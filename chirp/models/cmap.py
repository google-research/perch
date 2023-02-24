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

"""Metric for Class Mean Average Precision (CMAP)."""
from chirp.models import metrics
from clu import metrics as clu_metrics
import flax.struct
import jax
from jax import numpy as jnp


@flax.struct.dataclass
class CMAP(
    # TODO(bartvm): Create class factory for calculating over different outputs
    clu_metrics.CollectingMetric.from_outputs(("label", "label_logits"))
):
  """(Class-wise) mean average precision.

  This metric calculates the average precision score of each class, and also
  returns the average of those values. This is sometimes referred to as the
  macro-averaged average precision, or the class-wise mean average precision
  (CmAP).
  """

  def compute(self, sample_threshold: int = 0):
    values = super().compute()
    # Matrices can be large enough to make GPU/TPU run OOM, so use CPU instead
    with jax.default_device(jax.devices("cpu")[0]):
      mask = jnp.sum(values["label"], axis=0) > sample_threshold
      if jnp.sum(mask) == 0:
        return {"macro": 0.0}
      # Same as sklearn's average_precision_score(label, logits, average=None)
      # but that implementation doesn't scale to 10k+ classes
      class_aps = metrics.average_precision(
          values["label_logits"][:, mask].T, values["label"][:, mask].T
      )
      return {
          "macro": jnp.mean(class_aps),
          **{
              str(i): ap
              for i, ap in zip(
                  jnp.arange(values["label"].shape[1])[mask], class_aps
              )
          },
      }
