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

"""Evaluate a trained model."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from chirp import config_utils
from chirp.configs import config_globals
from chirp.eval import eval_lib
import jax
from ml_collections.config_flags import config_flags

_CONFIG = config_flags.DEFINE_config_file('config')
_EVAL_RESULTS_HEADER = (
    'eval_species',
    'average_precision',
    'roc_auc',
    'num_pos_match',
    'num_neg_match',
    'eval_set_name',
)
flags.mark_flags_as_required(['config'])


def _main():
  """Main function."""
  logging.info(_CONFIG.value)
  # We need to set Jax and TF GPU options before any other jax/tf calls.
  # Since calls can potentially happen in parse_config, we'll handle GPU options
  # before parsing the config.
  if hasattr(_CONFIG.value, 'jax_mem_frac'):
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(_CONFIG.value)
  if hasattr(_CONFIG.value, 'tf_gpu_growth') and _CONFIG.value.tf_gpu_growth:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

  config = config_utils.parse_config(
      _CONFIG.value, config_globals.get_globals()
  )

  # Check that the required user-specified fields are set in the config.
  if config.create_species_query is None:
    raise ValueError(
        'eval.py requires `config.create_species_query` to be set '
        'to a boolean value (True or False) in the passed config. '
        'Please update your config file and run again.'
    )
  if config.score_search is None:
    raise ValueError(
        'eval.py requires `config.score_search` to be set to a '
        'boolean value (True or False) in the passed config. '
        'Please update your config file and run again.'
    )
  if config.sort_descending is None:
    raise ValueError(
        'eval.py requires `sort_descending` to be set to a '
        'boolean value (True or False) in the passed config. '
        'Please update your config file and run again.'
    )
  # Ensure that every evaluation script includes an instantiation of a Pipeline
  # object with any desired data processing ops.
  for dataset_config in config.dataset_configs.values():
    if dataset_config.pipeline is None:
      raise ValueError(
          'eval.py requires each dataset_config in `config.dataset_configs` to '
          'have a `pipeline` attribute set to a '
          '`config_utils.callable_config` object with any desired data '
          'processing operations (ops).'
      )

  eval_datasets = eval_lib.load_eval_datasets(config)
  embedded_datasets = dict()
  for dataset_name, dataset in eval_datasets.items():
    logging.info('%s:\n%s', dataset_name, dataset)
    embedded_datasets[dataset_name] = eval_lib.get_embeddings(
        dataset, config.model_callback, config.batch_size
    )

  eval_set_search_results = dict()
  for eval_set in eval_lib.prepare_eval_sets(config, embedded_datasets):
    logging.info(eval_set.name)

    search_results = eval_lib.search(
        eval_set,
        config.model_callback.learned_representations,
        config.create_species_query,
        config.score_search,
    )

    eval_set_search_results[eval_set.name] = search_results

  # Collect eval set species performance results as a list of tuples.
  eval_metrics = [_EVAL_RESULTS_HEADER]
  for eval_set_name, eval_set_results in eval_set_search_results.items():
    eval_metrics.extend(
        eval_lib.compute_metrics(
            eval_set_name, eval_set_results, config.sort_descending
        )
    )

  # In a multi-host setting, only the first host should write results
  if jax.process_index() == 0:
    eval_lib.write_results_to_csv(
        eval_metrics, config.write_results_dir, config.write_filename
    )  # pytype: disable=wrong-arg-types  # jax-ndarray


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with jax.default_matmul_precision('float32'):
    _main()


if __name__ == '__main__':
  app.run(main)
