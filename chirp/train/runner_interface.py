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

"""Interface for jobs run by main.py."""

import dataclasses

from ml_collections import config_dict


@dataclasses.dataclass
class Runner:
  """Interface for jobs run by main.py.

  Users should strongly prefer putting any project-specific info in config.
  This allows easy hyper-parameter sweep, and avoids needing to update other
  job/project scripts.

  Attributes:
    mode: User-defined arbitrary string corresponding to the current job type.
      eg, 'train' vs 'eval' vs 'finetune'.
    workdir: Location for storing logs, checkpoints, etc.
    tf_data_service_address: Address of TF Data Service, for producing data
      batches faster.
  """

  mode: str
  workdir: str
  tf_data_service_address: str

  def run(self, config: config_dict.ConfigDict) -> None:
    raise NotImplementedError()
