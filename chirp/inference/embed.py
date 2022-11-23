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

"""Extract embeddings for a corpus of audio."""

import time
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.io.filesystem import CompressionTypes
from chirp import config_utils
from chirp.configs import config_globals
from chirp.configs.inference import raw_soundscapes
from chirp.inference import embed_lib
from etils import epath
import numpy as np

FLAGS = flags.FLAGS

_DRY_RUN_ONLY = flags.DEFINE_bool('dry_run', False,
                                  'Whether to execute a dry-run only.')


def dry_run(config, source_files):
  """Perform a dry run: check that the model loads and can process a file."""
  test_embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
  print('starting dry run....')
  test_embed_fn.setup()
  print('   loaded test model....')
  start = time.time()
  test_file = np.random.choice(source_files)
  print(f'   processing test file {test_file}')
  if test_embed_fn.embedding_model.window_size_s > 0:
    crop_s = 2 * test_embed_fn.embedding_model.window_size_s
  else:
    crop_s = 10
  got = test_embed_fn.process(
      embed_lib.SourceInfo(test_file, 0, 1), crop_s=crop_s)
  elapsed = time.time() - start
  if not got:
    raise Exception('Something went wrong; no results found.')
  test_embed_fn.teardown()
  print(f'Dry run successful! Party! Inference time : {elapsed:5.3f}')


def main(unused_argv: Sequence[str]) -> None:
  logging.info('Loading config')
  # TODO(tomdenton): Find a better config system that works for Beam workers.
  config = raw_soundscapes.get_config()
  config = config_utils.parse_config(config, config_globals.get_globals())

  logging.info('Locating source files...')
  source_files = []
  for pattern in config.source_file_patterns:
    for source_file in epath.Path('').glob(pattern):
      source_files.append(source_file)
  output_dir = epath.Path(config.output_dir)
  output_dir.mkdir(exist_ok=True)
  logging.info('Found %d source files.', len(source_files))
  if _DRY_RUN_ONLY.value:
    dry_run(config, source_files)
    return

  # Create and run the beam pipeline.
  source_infos = embed_lib.create_source_infos(source_files,
                                               config.num_shards_per_file)
  pipeline = beam.Pipeline()
  _ = (
      pipeline
      | beam.Create(source_infos)
      | beam.ParDo(embed_lib.EmbedFn(**config.embed_fn_config))
      # When a file is corrupted and can't be loaded EmbedFn
      # returns None. In this case the lambda below returns false, which then
      # filters it out.
      | beam.Filter(lambda x: x)
      | beam.io.tfrecordio.WriteToTFRecord(
          config.output_dir,
          compression_type=CompressionTypes.GZIP,
          file_name_suffix='.gz'))
  pipeline.run()


if __name__ == '__main__':
  app.run(main)
