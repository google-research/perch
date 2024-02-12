# coding=utf-8
# Copyright 2024 The Perch Authors.
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

r"""Convert files to embedding vector sstables.

Example command line:
python -m beam_index \
--source_files=test_files/*.wav \
--model_path=models/taxo_lori_34078450 \
--separation_model_path=models/separator4 \
--output_dir=lorikeet_inference \
--hints_tag=eaus \
"""

import collections
import gc
import glob
import os
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import data_tools
import model_utils
import numpy as np
import taxonomy
import tensorflow as tf

FLAGS = flags.FLAGS
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SPECIES_INFO_PATH = os.path.join(DATA_PATH, 'species_info.csv')
ENSEMBLE_SIZE = 3

flags.DEFINE_list('source_files', [], 'Source audio files (wav or mp3).')
flags.DEFINE_string(
    'model_path', '', 'Where to find the model params and inference.pb'
)
flags.DEFINE_string(
    'separation_model_path', '', 'Where to find the separation inference model.'
)
flags.DEFINE_string('output_dir', '', 'Where to dump output data.')
flags.DEFINE_string(
    'embedding_key', 'hidden_embedding', 'Embedding output key in saved_model.'
)
flags.DEFINE_integer(
    'file_shards', 48, 'Number of sub-jobs to divide each input file into.'
)
flags.DEFINE_string('hints_tag', '', 'Species set tag for hints.')
flags.DEFINE_boolean('dry_run', False, 'Whether to exit after dry-run.')

PredictionTuple = collections.namedtuple(
    'PredictionTuple', ['file_id', 'start_time', 'end_time', 'logits']
)


class EmbedFn(beam.DoFn):
  """Beam function for model inference."""

  def __init__(
      self,
      model_path,
      embedding_key,
      separation_model_path=None,
      sample_rate=22050,
      hints_tag=None,
  ):
    # Get a local copy of the inference.pb file.
    self.model_path = model_path
    self.separation_model_path = separation_model_path
    self.inference_path = os.path.join(model_path, 'inference')
    self.embedding_key = embedding_key
    self.sample_rate = sample_rate
    self.hints_tag = hints_tag

  def setup(self):
    # tf.compat.v1.disable_eager_execution()
    # admittedly a bit brittle...
    self.model_params = model_utils.load_params_from_json(self.model_path)
    self.taxo = taxonomy.Taxonomy(self.model_path, DATA_PATH, SPECIES_INFO_PATH)
    self.taxo.PrintEnumSizes()
    self.hints = self.taxo.MakeSpeciesHints(species_list_tag=self.hints_tag)

    if self.separation_model_path:
      self.separation_model = model_utils.load_separation_model(
          self.separation_model_path
      )
    classifiers = model_utils.load_classifier_ensemble(
        self.model_path, max_runs=1
    )
    self.embedding_model = list(classifiers.values())[0]

  def get_hints(self, batch_size):
    if self.hints is not None:
      hints = self.hints[np.newaxis, :]
      hints = np.tile(hints, [batch_size, 1])
    else:
      hints = np.ones([batch_size, self.taxo.NumLabels()])
    return hints

  def embed(self, file_id, audio, timestamp_offset):
    """Convert target audio to embeddings."""
    window_size_s = self.model_params.dataset.window_size_s
    hop_size_s = window_size_s / 2
    logging.info('...starting separation (%s)', file_id)
    sep_chunks, raw_chunks = model_utils.separate_windowed(
        audio,
        self.separation_model,
        hop_size_s,
        window_size_s,
        self.sample_rate,
    )
    raw_chunks = raw_chunks[:, np.newaxis, :]
    stacked_chunks = np.concatenate([raw_chunks, sep_chunks], axis=1)
    n_chunks = stacked_chunks.shape[0]
    n_channels = stacked_chunks.shape[1]
    big_batch = np.reshape(stacked_chunks, [n_chunks * n_channels, -1])
    # We often get memory blowups at this point; trigger a garbage collection.
    gc.collect()

    logging.info('...creating embeddings (%s)', file_id)
    embedding = model_utils.model_embed(
        big_batch,
        self.embedding_model,
        hints=self.get_hints(big_batch.shape[0]),
        output_key=self.embedding_key,
    )
    embedding = np.reshape(embedding, [n_chunks, n_channels, -1])
    print('embedding shape : ', embedding.shape)

    serialized_embedding = tf.io.serialize_tensor(embedding)
    feature = {
        'file_id': data_tools.BytesFeature(bytes(file_id, encoding='utf8')),
        'timestamp_offset': data_tools.IntFeature(timestamp_offset),
        'embedding': data_tools.BytesFeature(serialized_embedding.numpy()),
        'embedding_shape': data_tools.IntFeature(embedding.shape),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    beam.metrics.Metrics.counter('beaminference', 'segments_processed').inc()
    return [ex]

  @beam.typehints.with_output_types(Any)
  def process(self, source_info, crop_s=-1):
    audio_filepath, shard_num, num_shards = source_info
    file_name = os.path.basename(audio_filepath)
    file_id = file_name.split('.')[0]

    try:
      logging.info('...loading audio (%s)', audio_filepath)
      audio, _ = data_tools.LoadAudio(audio_filepath, self.sample_rate)
    except Exception as e:  # pylint: disable=broad-except
      beam.metrics.Metrics.counter('beaminference', 'load_audio_error').inc()
      logging.error('Failed to load audio : %s', audio_filepath)
      logging.exception('Load audio exception : %s', e)
      return

    if audio.shape[0] < 2 * self.model_params.dataset.window_size:
      beam.metrics.Metrics.counter('beaminference', 'short_audio_error').inc()
      logging.error('short audio file : %s', audio_filepath)
      return

    if num_shards > 1:
      shard_len = audio.shape[0] // num_shards
      timestamp_offset = shard_num * shard_len
      audio = audio[timestamp_offset : timestamp_offset + shard_len]
    else:
      timestamp_offset = 0

    if crop_s > 0:
      audio = audio[: crop_s * self.sample_rate]
    return self.embed(file_id, audio, timestamp_offset)


def get_counter(metrics, name):
  counter = metrics.query(beam.metrics.MetricsFilter().with_name(name))[
      'counters'
  ]
  if not counter:
    return 0
  return counter[0].result


def main(unused_argv):
  source_files = []
  for pattern in FLAGS.source_files:
    source_files += glob.glob(pattern)

  print('Found %d source files.' % len(source_files))

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  source_file_splits = []
  for s in source_files:
    for i in range(FLAGS.file_shards):
      source_file_splits.append((s, i, FLAGS.file_shards))

  # Dry-run.
  print('Starting dry run...')
  test_fn = EmbedFn(
      FLAGS.model_path,
      FLAGS.embedding_key,
      FLAGS.separation_model_path,
      hints_tag=FLAGS.hints_tag,
  )
  test_fn.setup()
  got_results = False
  start = time.time()
  print(source_file_splits[15])
  for unused_p in test_fn.process(source_file_splits[15], crop_s=10):
    got_results = True
  elapsed = time.time() - start
  if not got_results:
    raise Exception('Something went wrong; no results found.')
  test_fn.teardown()
  print('Dry run successful! Party! Inference time : %5.3f' % elapsed)
  if FLAGS.dry_run:
    return
  output_prefix = os.path.join(FLAGS.output_dir, 'embeddings')
  pipeline = beam.Pipeline()
  _ = (
      pipeline
      | beam.Create(source_file_splits)
      | beam.ParDo(
          EmbedFn(
              FLAGS.model_path,
              FLAGS.embedding_key,
              FLAGS.separation_model_path,
              hints_tag=FLAGS.hints_tag,
          )
      )
      # When a file is corrupted and can't be loaded InferenceFn
      # returns None. In this case the lambda below returns false, which then
      # filters it out.
      | beam.Filter(lambda x: x)
      | beam.io.WriteToTFRecord(
          output_prefix, coder=beam.coders.ProtoCoder(tf.train.Example)
      )
  )
  pipeline.run()


if __name__ == '__main__':
  app.run(main)
