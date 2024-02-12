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

"""Extract embeddings for a corpus of audio."""

import time
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from chirp import config_utils
from chirp.configs import config_globals
from chirp.inference import embed_lib
from etils import epath
import numpy as np

FLAGS = flags.FLAGS

_CONFIG_KEY = flags.DEFINE_string(
    'config', 'raw_soundscapes', 'Name of the config to use.'
)
_DRY_RUN_ONLY = flags.DEFINE_bool(
    'dry_run', False, 'Whether to execute a dry-run only.'
)
_DRY_RUN_CROP_S = flags.DEFINE_float(
    'dry_run_crop_s', 10.0, 'Amount of audio to use for dry run.'
)


def dry_run(config, source_infos):
  """Perform a dry run: check that the model loads and can process a file."""
  test_embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
  print('starting dry run....')
  test_embed_fn.setup()
  print('   loaded test model....')
  start = time.time()
  test_source = np.random.choice(source_infos)
  print(f'   processing test source {test_source}')
  got = test_embed_fn.process(test_source, _DRY_RUN_CROP_S.value)
  elapsed = time.time() - start
  if not got:
    # pylint: disable=broad-exception-raised
    raise ValueError('Something went wrong; no results found.')
  test_embed_fn.teardown()
  print(f'Dry run successful! Party! Inference time : {elapsed:5.3f}')


def main(unused_argv: Sequence[str]) -> None:
  logging.info('Loading config')
  config = embed_lib.get_config(_CONFIG_KEY.value)
  config = config_utils.parse_config(config, config_globals.get_globals())

  logging.info('Locating source files...')
  # Create and run the beam pipeline.
  source_infos = embed_lib.create_source_infos(
      source_file_patterns=config.source_file_patterns,
      shard_len_s=config.get('shard_len_s', -1.0),
      num_shards_per_file=config.get('num_shards_per_file', -1),
      start_shard_idx=config.get('start_shard_idx', 0),
  )
  logging.info('Found %d source infos.', len(source_infos))
  if not source_infos:
    raise ValueError('No source infos found.')

  if _DRY_RUN_ONLY.value:
    dry_run(config, source_infos)
    return

  output_dir = epath.Path(config.output_dir)
  output_dir.mkdir(exist_ok=True, parents=True)
  embed_lib.maybe_write_config(config, output_dir)

  options = beam.options.pipeline_options.PipelineOptions(
      runner='DirectRunner',
      direct_num_workers=config.num_direct_workers,
      direct_running_mode='in_memory')
  pipeline = beam.Pipeline(options=options)
  embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
  embed_lib.build_run_pipeline(
      pipeline, config.output_dir, source_infos, embed_fn
  )


if __name__ == '__main__':
  app.run(main)
