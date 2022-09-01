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
  config = config_utils.parse_config(_CONFIG.value,
                                     config_globals.get_globals())

  eval_datasets = eval_lib.load_eval_datasets(config)
  for dataset_name, dataset in eval_datasets.items():
    logging.info('%s:\n%s', dataset_name, dataset)


if __name__ == '__main__':
  app.run(main)
