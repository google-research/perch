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
from typing import Any

from chirp.models import metrics
from clu import metrics as clu_metrics
import flax
from jax import numpy as jnp


@flax.struct.dataclass
class CMAP(clu_metrics.Metric):
  """Class Mean Average Precision metric.

  Accumulates logits and labels to allow computing MAP per-class across the
  full dataset. See the definition here:
  https://www.imageclef.org/BirdCLEF2019

  Caution: This implementation does not work in jit'ed functions, because
  concatenation of scores and labels has non-static shape. Workarounds for this
  exist, but are very ugly. Alternatively, a streaming approximation is
  possible, similar to the Keras implementation of AUC.

  Thus, it is recommended to compute CMAP metrics by accumulating scores
  and labels outside of the jit'ed loop.

  Attributes:
    scores: An array of logits or scores, with shape [batch, num_classes].
    labels: Array of ground-truth labels with shape [batch, num_classes].
  """

  scores: jnp.ndarray | None = None
  labels: jnp.ndarray | None = None

  @classmethod
  def empty(cls) -> "CMAP":
    return cls(scores=None, labels=None)

  @classmethod
  def from_model_output(
      cls, values: tuple[jnp.array, jnp.array], **_
  ) -> clu_metrics.Metric:
    scores, labels = values
    return cls(scores=scores, labels=labels)

  def merge(self, other):
    if self.scores is None:
      return other
    if other.scores is None:
      return self
    if other.scores.ndim not in [2, 3]:
      raise ValueError(
          "Expecting the scores to be in one of the following"
          "formats: [n_devices, batch_size, n_classes] or"
          "[batch_size, n_classes]. Current shape is"
          f"{self.scores.shape}"
      )
    return type(self)(
        scores=jnp.concatenate((self.scores, other.scores), axis=-2),
        labels=jnp.concatenate((self.labels, other.labels), axis=-2),
    )

  def compute(self, class_wise=False, sample_threshold: int = 0) -> Any:
    """Compute cmap only using classes that have > sample_threshold samples."""
    # Compute average precision over the batch axis.
    class_av_prec = metrics.average_precision(self.scores.T, self.labels.T)
    class_counts = jnp.sum(self.labels, axis=0, keepdims=True)
    # Mask classes with no labels to avoid inflating the CMAP.
    class_av_prec *= class_counts > sample_threshold
    if class_wise:
      return jnp.squeeze(class_av_prec)
    return jnp.sum(class_av_prec) / jnp.sum(class_counts > sample_threshold)


def make_cmap_metrics_dict(label_names):
  """Create a dict of empty cmap_metrics."""
  return {label: CMAP.empty() for label in label_names}


def update_cmap_metrics_dict(cmap_metrics, model_outputs, batch):
  """Update a dict of cmap_metrics from model_outputs and a batch."""
  for label_name in cmap_metrics:
    cmap_metrics[label_name] = cmap_metrics[label_name].merge(
        CMAP(getattr(model_outputs, label_name), batch[label_name])
    )
  return cmap_metrics
