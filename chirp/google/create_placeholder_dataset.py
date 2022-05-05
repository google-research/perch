# coding=utf-8
# Copyright 2022 The Chirp Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Creates a TFDS-compatible placeholder dataset on CNS for the SN taxonomy.

Usage (with Placer replication):

placer prepare --overwrite \
  /placer/prod/scratch/home/kakapo/public_data/placeholder
fileutil mkdir --gfs_user=kakapo \
  /placer/prod/scratch/home/kakapo/public_data/placeholder
blaze run -c opt //third_party/py/kakapo/google:create_placeholder_dataset -- \
  --gfs_user=kakapo
placer publish /placer/prod/scratch/home/kakapo/public_data/placeholder
"""

import json
import os
import tempfile
import time
from typing import Sequence

from absl import app
from absl import flags
from etils import epath
import tensorflow as tf
import tensorflow_datasets as tfds

from google3.file.util.effingo.public import effingo_pb2
from google3.file.util.effingo.public import pywrapeffingoclient as effingo
from google3.pyglib import gfile
from google3.pyglib.concurrent import parallel
from google3.sstable.python import sstable

_CNS_SOURCE_DIR = '/cns/is-d/home/bioacoustics/rs=6.4/BirdCLEF2019/taxo_sn/'
_X20_SOURCE_DIR = '/google/data/rw/users/to/tomdenton/birdsep'
_PLACER_DEST_DIR = '/placer/prod/scratch/home/kakapo/public_data/placeholder'


def copy_and_add_label_str(source_path: str, destination_path: str,
                           destination_features: tfds.features.FeaturesDict,
                           label_set: Sequence[str]) -> None:
  """Copies an SSTable of tf.Example and adds a 'label_str' feature."""
  # The source tf.Example protos don't have a 'label_str' field.
  source_features = tfds.features.FeaturesDict(
      {k: v for k, v in destination_features.items() if k != 'label_str'})

  with tempfile.TemporaryDirectory() as temp_dir:
    # Work locally for a faster throughput.
    tmp_source_path = os.path.join(temp_dir, 'source.sstable')
    tmp_destination_path = os.path.join(temp_dir, 'destination.sstable')
    gfile.Copy(source_path, tmp_source_path)

    input_table = sstable.SSTable(tmp_source_path)
    with sstable.Builder(tmp_destination_path) as builder:
      for k, v in input_table.items():
        # Add 'label_str' feature.
        f = {
            k: v.numpy()  # pytype: disable=attribute-error
            for (k, v) in source_features.deserialize_example(v).items()
        }
        f['label_str'] = [label_set[label] for label in f['label']]

        builder.Add(k, v)

    gfile.Copy(tmp_destination_path, destination_path, overwrite=True)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Load dataset info and instantiate FeaturesDict.
  with gfile.Open(os.path.join(_CNS_SOURCE_DIR, 'info.json'), 'r') as f:
    info = json.load(f)

  features = tfds.features.FeaturesDict({
      'audio':
          tfds.features.Audio(
              dtype=tf.float32, sample_rate=info['sample_rate_hz']),
      'label':
          tfds.features.Sequence(
              tfds.features.ClassLabel(names=info['label_set'])),
      'label_str':
          tfds.features.Sequence(tfds.features.Text()),
      'bg_labels':
          tfds.features.Sequence(
              tfds.features.ClassLabel(names=info['label_set'])),
      'genus':
          tfds.features.Sequence(
              tfds.features.ClassLabel(names=info['genus_set'])),
      'family':
          tfds.features.Sequence(
              tfds.features.ClassLabel(names=info['family_set'])),
      'order':
          tfds.features.Sequence(
              tfds.features.ClassLabel(names=info['order_set'])),
      'filename':
          tfds.features.Text(),
  })

  # Copy training and validation SSTables to destination directory.
  source_destination_pairs = []
  for source_subdir, split in zip(('train_xc', 'eval_xc'), ('train', 'valid')):
    for source_path in gfile.Glob(
        os.path.join(_CNS_SOURCE_DIR, source_subdir, 'tf.sstable-*')):
      source_destination_pairs.append(
          (source_path,
           os.path.join(
               _PLACER_DEST_DIR,
               os.path.basename(source_path).replace(
                   'tf.sstable', f'placeholder-{split}.sstable'))))

  client = effingo.NewEffingoClient(effingo.EffingoClientConfig())
  client.WaitUntilInitialized(time.time() + 5 * 60)

  copy_settings = effingo.CopySettings()
  copy_settings.user = flags.FLAGS.gfs_user
  copy_settings.priority = 200
  copy_settings.colossus_source_options.request_priority = effingo_pb2.TRANSFER
  copy_settings.colossus_destination_options.request_priority = effingo_pb2.TRANSFER
  copy_settings.overwrite = True

  client.CopyFilesAndWait(source_destination_pairs, copy_settings)

  # Copy test SSTables and add 'label_str' feature.
  x20_source_paths = [
      os.path.join(_X20_SOURCE_DIR, source_filename)
      for source_filename in ('caples_eval_22050.sst',
                              'high_sierras_eval_22050.sst')
  ]
  x20_destination_paths = [
      os.path.join(_PLACER_DEST_DIR,
                   f'placeholder-{split}.sstable-00000-of-00001')
      for split in ('test_caples', 'test_high_sierras')
  ]
  kwargs_list = []
  for s, d in zip(x20_source_paths, x20_destination_paths):
    kwargs_list.append({
        'source_path': s,
        'destination_path': d,
        'destination_features': features,
        'label_set': info['label_set']
    })

  parallel.RunInParallel(
      copy_and_add_label_str, kwargs_list, num_workers=len(kwargs_list))

  # Write dataset metadata to destination directory.
  tfds.folder_dataset.write_metadata(
      data_dir=_PLACER_DEST_DIR,
      features=features,
      split_infos=tfds.folder_dataset.compute_split_info(
          filename_template=tfds.core.ShardedFileTemplate(
              data_dir=epath.Path(_PLACER_DEST_DIR),
              dataset_name='placeholder',
              filetype_suffix='sstable')),
      metadata=tfds.core.MetadataDict(example_size_s=info['example_size_s']),
      description="""Placeholder taxonomy dataset.""",
      homepage='https://github.com/google-research/chirp',
      supervised_keys=('audio', 'label'),
      citation="""Internal.""")


if __name__ == '__main__':
  app.run(main)
