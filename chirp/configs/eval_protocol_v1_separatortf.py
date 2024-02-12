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

"""Sample evaluation protocol v1 configuration."""

from chirp import config_utils
from chirp.configs import eval_protocol_v1_base
from ml_collections import config_dict

_c = config_utils.callable_config
_object_config = config_utils.object_config

SEP_PATH = ''


def get_config() -> config_dict.ConfigDict:
  """Creates a configuration dictionary for the evaluation protocol v1."""
  config = eval_protocol_v1_base.get_config()

  # The model_callback is expected to be a Callable[[np.ndarray], np.ndarray].
  model_checkpoint_path = config_dict.FieldReference(SEP_PATH)
  config.model_checkpoint_path = model_checkpoint_path
  config.model_callback = _c(
      'eval_lib.SeparatorTFCallback', model_path=model_checkpoint_path
  )


  # TODO(bringingjoy): extend create_species_query to support returning multiple
  # queries for a given eval species.
  config.create_species_query = _object_config('eval_lib.create_averaged_query')
  config.score_search = _object_config('eval_lib.cosine_similarity')
  config.sort_descending = True

  return config
