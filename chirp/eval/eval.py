# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""[WIP] Evaluate a trained model."""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
from chirp import config_utils
from chirp.configs import config_globals
from chirp.eval import eval_lib
from ml_collections.config_flags import config_flags

_CONFIG = config_flags.DEFINE_config_file('config')
flags.mark_flags_as_required(['config'])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info(_CONFIG.value)
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
  for dataset_config in config.dataset_configs:
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
  for eval_set_name, eval_set_generator in eval_lib.prepare_eval_sets(
      config, embedded_datasets
  ):
    logging.info(eval_set_name)

    search_results = eval_lib.search(
        eval_set_generator,
        config.model_callback.learned_representations,
        config.create_species_query,
        config.score_search,
    )

    eval_set_search_results[eval_set_name] = search_results

  # Collect eval set species performance results as a list of tuples.
  eval_metrics = [('eval_species', 'average_precision', 'eval_set_name')]
  for eval_set_name, eval_set_results in eval_set_search_results.items():
    eval_metrics.extend(
        eval_lib.compute_metrics(
            eval_set_name, eval_set_results, config.sort_descending
        )
    )

  eval_lib.write_results_to_csv(eval_metrics, config.write_results_dir)  # pytype: disable=wrong-arg-types  # jax-ndarray


if __name__ == '__main__':
  app.run(main)
