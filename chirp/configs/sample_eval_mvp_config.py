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

"""Sample MVP evaluation configuration."""

from chirp import config_utils
from chirp.configs import baseline_attention
from chirp.configs import eval_mvp_base
from ml_collections import config_dict

_c = config_utils.callable_config
_object_config = config_utils.object_config


def get_config() -> config_dict.ConfigDict:
  """Creates a configuration dictionary for the MVP evaluation protocol."""
  config = eval_mvp_base.get_config()
  baseline_attention_config = baseline_attention.get_config()

  # The model_callback is expected to be a Callable[[np.ndarray], np.ndarray].
  model_checkpoint_path = config_dict.FieldReference('')
  config.model_checkpoint_path = model_checkpoint_path
  config.model_callback = _c(
      'eval_lib.TaxonomyModelCallback',
      init_config=baseline_attention_config.init_config,
      workdir=model_checkpoint_path)


  # TODO(bringingjoy): extend create_species_query to support returning multiple
  # queries for a given eval species.
  config.create_species_query = _object_config('eval_lib.create_averaged_query')
  config.score_search = _object_config('eval_lib.cosine_similarity')
  config.score_search_ordering = 'high'

  return config
