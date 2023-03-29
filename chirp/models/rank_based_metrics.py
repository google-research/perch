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

"""Rank-based metrics, including cmAP and generalized mean rank."""
from chirp.models import metrics
from clu import metrics as clu_metrics
import flax.struct
import jax
from jax import numpy as jnp


@flax.struct.dataclass
class RankBasedMetrics(
    # TODO(bartvm): Create class factory for calculating over different outputs
    clu_metrics.CollectingMetric.from_outputs(("label", "label_logits"))
):
  """(Class-wise) rank-based metrics.

  This metric calculates the average precision score of each class, and also
  returns the average of those values. This is sometimes referred to as the
  macro-averaged average precision, or the class-wise mean average precision
  (CmAP).

  It also calculates the generalized mean rank for each class and returns the
  geometric average of those values.
  """

  def compute(self, sample_threshold: int = 0):
    values = super().compute()
    # Matrices can be large enough to make GPU/TPU run OOM, so use CPU instead
    with jax.default_device(jax.devices("cpu")[0]):
      mask = jnp.sum(values["label"], axis=0) > sample_threshold
      if jnp.sum(mask) == 0:
        return {"macro_cmap": 0.0, "macro_gmr": 0.0}
      # Same as sklearn's average_precision_score(label, logits, average=None)
      # but that implementation doesn't scale to 10k+ classes
      class_aps = metrics.average_precision(
          values["label_logits"].T, values["label"].T
      )
      class_aps = jnp.where(mask, class_aps, jnp.nan)

      class_gmr, class_gmr_var = metrics.generalized_mean_rank(
          values["label_logits"].T, values["label"].T
      )
      class_gmr = jnp.where(mask, class_gmr, jnp.nan)
      class_gmr_var = jnp.where(mask, class_gmr_var, jnp.nan)
      class_num_tp = jnp.where(
          mask, jnp.sum(values["label"] > 0, axis=0), jnp.nan
      )

      return {
          "macro_cmap": jnp.mean(class_aps, where=mask),
          "individual_cmap": class_aps,
          # If the GMR is 0.0 for at least one class, then the geometric average
          # goes to zero. Instead, we take the geometric average of 1 - GMR and
          # then take 1 - geometric_average.
          "macro_gmr": 1.0 - jnp.exp(
              jnp.mean(jnp.log(1.0 - class_gmr), where=mask)
          ),
          "individual_gmr": class_gmr,
          "individual_gmr_var": class_gmr_var,
          "individual_num_tp": class_num_tp,
      }


def add_rank_based_metrics_to_metrics_collection(name: str, metrics_collection):
  """Adds a RankBasedMetric instance to an existing CLU metrics collection."""
  new_collection = flax.struct.dataclass(
      type(
          "_ValidCollection",
          (metrics_collection,),
          {
              "__annotations__": {
                  f"{name}_rank_based": RankBasedMetrics,
                  **metrics_collection.__annotations__,
              }
          },
      )
  )
  return new_collection
