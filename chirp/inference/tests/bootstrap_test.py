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

"""Tests for project state handling."""

import os
import shutil
import tempfile

from chirp import audio_utils
from chirp.inference import embed_lib
from chirp.inference import tf_examples
from chirp.inference.search import bootstrap
from chirp.inference.search import search
from etils import epath
from ml_collections import config_dict
import numpy as np
from scipy.io import wavfile

from absl.testing import absltest


class BootstrapTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def make_wav_files(self, classes, filenames):
    # Create a pile of files.
    rng = np.random.default_rng(seed=42)
    for subdir in classes:
      subdir_path = os.path.join(self.tempdir, subdir)
      os.mkdir(subdir_path)
      for filename in filenames:
        with open(
            os.path.join(subdir_path, f'{filename}_{subdir}.wav'), 'wb'
        ) as f:
          noise = rng.normal(scale=0.2, size=16000)
          wavfile.write(f, 16000, noise)
    audio_glob = os.path.join(self.tempdir, '*/*.wav')
    return audio_glob

  def write_placeholder_embeddings(self, audio_glob, source_infos, embed_dir):
    """Utility method for writing embeddings with the placeholder model."""
    # Set up embedding function.
    config = config_dict.ConfigDict()
    model_kwargs = {
        'sample_rate': 16000,
        'embedding_size': 128,
        'make_embeddings': True,
        'make_logits': False,
        'make_separated_audio': False,
        'window_size_s': 5.0,
        'hop_size_s': 5.0,
    }
    embed_fn_config = config_dict.ConfigDict()
    embed_fn_config.write_embeddings = True
    embed_fn_config.write_logits = False
    embed_fn_config.write_separated_audio = False
    embed_fn_config.write_raw_audio = False
    embed_fn_config.write_frontend = False
    embed_fn_config.model_key = 'placeholder_model'
    embed_fn_config.model_config = model_kwargs
    embed_fn_config.min_audio_s = 0.1
    embed_fn_config.file_id_depth = 1
    config.embed_fn_config = embed_fn_config
    config.source_file_patterns = [audio_glob]
    config.num_shards_per_file = -1
    config.shard_len_s = 5

    embed_lib.maybe_write_config(config, epath.Path(embed_dir))

    embed_fn = embed_lib.EmbedFn(**embed_fn_config)
    embed_fn.setup()

    # Write embeddings.
    audio_loader = lambda fp, offset: audio_utils.load_audio_window(
        fp, offset, sample_rate=16000, window_size_s=5.0
    )
    audio_iterator = audio_utils.multi_load_audio_window(
        filepaths=[s.filepath for s in source_infos],
        offsets=[s.shard_num * s.shard_len_s for s in source_infos],
        audio_loader=audio_loader,
    )
    with tf_examples.EmbeddingsTFRecordMultiWriter(
        output_dir=embed_dir, num_files=1
    ) as file_writer:
      for source_info, audio in zip(source_infos, audio_iterator):
        file_id = source_info.file_id(1)
        offset_s = source_info.shard_num * source_info.shard_len_s
        example = embed_fn.audio_to_example(file_id, offset_s, audio)
        file_writer.write(example.SerializeToString())
      file_writer.flush()

  def test_filesystem_source_map(self):
    """Check consistency of source_map function and SourceInfo file ids."""
    classes = ['pos', 'neg']
    filenames = ['foo', 'bar', 'baz']
    audio_glob = self.make_wav_files(classes, filenames)
    source_infos = embed_lib.create_source_infos([audio_glob], shard_len_s=5.0)
    self.assertLen(source_infos, len(classes) * len(filenames))

    for fid_depth in (1, 2):
      source_map = lambda file_id, offset: bootstrap.filesystem_source_map(
          audio_globs=[audio_glob],
          file_id_depth=fid_depth,  # pylint: disable=cell-var-from-loop
          file_id=file_id,
      )
      for source_info in source_infos:
        file_id = source_info.file_id(file_id_depth=fid_depth)
        got_filepath = source_map(file_id, 0.0)
        self.assertTrue(epath.Path(got_filepath).exists())
      # Check that we don't find non-existent files.
      self.assertRaises(ValueError, source_map, 'nonexistent.wav', 0.0)

  def test_bootstrap_from_embeddings(self):
    classes = ['pos', 'neg']
    filenames = ['foo', 'bar', 'baz']
    audio_glob = self.make_wav_files(classes, filenames)
    source_infos = embed_lib.create_source_infos([audio_glob], shard_len_s=5.0)
    self.assertLen(source_infos, len(classes) * len(filenames))

    embed_dir = os.path.join(self.tempdir, 'embeddings')
    labeled_dir = os.path.join(self.tempdir, 'labeled')
    epath.Path(embed_dir).mkdir(parents=True, exist_ok=True)
    epath.Path(labeled_dir).mkdir(parents=True, exist_ok=True)

    self.write_placeholder_embeddings(audio_glob, source_infos, embed_dir)

    bootstrap_config = bootstrap.BootstrapConfig.load_from_embedding_path(
        embeddings_path=embed_dir,
        annotated_path=labeled_dir,
    )
    print('config hash : ', bootstrap_config.embedding_config_hash())

    project_state = bootstrap.BootstrapState(
        config=bootstrap_config,
    )

    # Check that we can iterate over the embeddings and get the correct number.
    ds = project_state.create_embeddings_dataset()
    got_embeds = [ex for ex in ds.as_numpy_iterator()]
    self.assertLen(got_embeds, len(source_infos))

    # Check that we can iterate over TopKSearchResults,
    # and successfully attach audio.
    search_results = search.TopKSearchResults(top_k=3)
    for i, ex in enumerate(ds.as_numpy_iterator()):
      result = search.SearchResult(
          embedding=ex['embedding'],
          score=1.0 * i,
          sort_score=1.0 * i,
          filename=str(ex['filename'], 'utf-8'),
          timestamp_offset=float(ex['timestamp_s']),
      )
      search_results.update(result)
    audio_iterator = project_state.search_results_audio_iterator(search_results)
    for result in audio_iterator:
      self.assertEqual(result.audio.shape, (16000,))


if __name__ == '__main__':
  absltest.main()
