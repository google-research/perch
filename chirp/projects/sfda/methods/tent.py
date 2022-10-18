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

"""Test-Time Entropy Minimization (TENT) method."""

from typing import Type
from chirp.projects.sfda import adapt
from chirp.projects.sfda import losses
from clu import metrics as clu_metrics


class Tent(adapt.SFDAMethod):
  """Test-time entropy minimization method."""

  _CITATION = ('Wang, Dequan, et al. "Tent: Fully test-time adaptation by '
               'entropy minimization." ICLR (2021).')

  def get_adaptation_metrics(self, supervised: bool, multi_label: bool,
                             **method_kwargs) -> Type[clu_metrics.Collection]:
    """Obtain metrics that will be monitored during adaptation."""

    metrics_dict = vars(
        adapt.get_common_metrics(
            supervised=supervised, multi_label=multi_label))['__annotations__']
    if multi_label:
      entropy_fn = losses.label_binary_ent
    else:
      entropy_fn = losses.label_ent
    metrics_dict['main_loss'] = clu_metrics.Average.from_fun(entropy_fn)
    return clu_metrics.Collection.create(**metrics_dict)
