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

"""A baseline that simply updates BatchNorm's statistics."""
from chirp.projects.sfda import adapt
from clu import metrics as clu_metrics


class AdaBN(adapt.SFDAMethod):
  """A baseline that simply updates BatchNorm's statistics.

  No optimization takes place, the method only fowards the data through the
  model, with the option 'update_bn_statistics' activated.
  """

  _CITATION = (
      "Li, Yanghao, et al. 'Revisiting batch normalization for practical "
      "domain adaptation.' arXiv preprint arXiv:1603.04779 (2016)."
  )

  def get_adaptation_metrics(
      self, supervised: bool, multi_label: bool, **method_kwargs
  ) -> type[clu_metrics.Collection]:
    """Obtain metrics that will be monitored during adaptation."""
    metrics_dict = vars(
        adapt.get_common_metrics(supervised=supervised, multi_label=multi_label)
    )["__annotations__"]

    return clu_metrics.Collection.create(**metrics_dict)
