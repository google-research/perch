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
from chirp import config_utils
from chirp.configs import config_globals
from chirp.configs.inference import birdnet_soundscapes
from chirp.configs.inference import raw_soundscapes
from chirp.configs.inference import separate_soundscapes
from chirp.inference import embed_lib
from etils import epath
import numpy as np

FLAGS = flags.FLAGS

_CONFIG_KEY = flags.DEFINE_string('config', 'raw_soundscapes',
                                  'Name of the config to use.')
_DRY_RUN_ONLY = flags.DEFINE_bool('dry_run', False,
                                  'Whether to execute a dry-run only.')
_DRY_RUN_CROP_S = flags.DEFINE_float('dry_run_crop_s', 10.0,
                                     'Amount of audio to use for dry run.')


def dry_run(config, source_files):
  """Perform a dry run: check that the model loads and can process a file."""
  test_embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
  print('starting dry run....')
  test_embed_fn.setup()
  print('   loaded test model....')
  start = time.time()
  test_file = np.random.choice(source_files)
  print(f'   processing test file {test_file}')
  got = test_embed_fn.process(
      embed_lib.SourceInfo(test_file, 0, 1), _DRY_RUN_CROP_S.value)
  elapsed = time.time() - start
  if not got:
    raise Exception('Something went wrong; no results found.')
  test_embed_fn.teardown()
  print(f'Dry run successful! Party! Inference time : {elapsed:5.3f}')


def main(unused_argv: Sequence[str]) -> None:
  logging.info('Loading config')
  # TODO(tomdenton): Find a better config system that works for Beam workers.
  if _CONFIG_KEY.value == 'birdnet_soundscapes':
    config = birdnet_soundscapes.get_config()
  elif _CONFIG_KEY.value == 'raw_soundscapes':
    config = raw_soundscapes.get_config()
  elif _CONFIG_KEY.value == 'separate_soundscapes':
    config = separate_soundscapes.get_config()
  else:
    raise ValueError('Unknown config.')
  config = config_utils.parse_config(config, config_globals.get_globals())

  logging.info('Locating source files...')
  source_files = []
  for pattern in config.source_file_patterns:
    for source_file in epath.Path('').glob(pattern):
      source_files.append(source_file)
  output_dir = epath.Path(config.output_dir)
  output_dir.parent.mkdir(exist_ok=True)
  logging.info('Found %d source files.', len(source_files))
  if _DRY_RUN_ONLY.value:
    dry_run(config, source_files)
    return

  # Create and run the beam pipeline.
  source_infos = embed_lib.create_source_infos(source_files,
                                               config.num_shards_per_file)
  pipeline = beam.Pipeline()
  embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
  embed_lib.build_run_pipeline(pipeline, config.output_dir, source_infos,
                               embed_fn)


if __name__ == '__main__':
  app.run(main)
