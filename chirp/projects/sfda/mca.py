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

"""Metric for Mean Class Accuracy (MCA)."""
from typing import Any, Optional

from clu import metrics as clu_metrics
import flax
from jax import numpy as jnp
import numpy as np


@flax.struct.dataclass
class MCA(clu_metrics.Metric):
  """Mean Class Accuracy metric.

  Accumulates information from individual batches to compute the mean class
  accuracy over the dataset. Specifically, this involves computing the per-class
  accuracy for each class in the dataset, and then averaging those.

  Attributes:
    scores: An array of logits or scores, with shape [batch, num_classes].
    labels: Array of ground-truth labels with shape [batch, num_classes].
  """
  counts_correct: Optional[jnp.array] = None
  counts_total: Optional[jnp.array] = None

  @classmethod
  def empty(cls) -> "MCA":
    return cls(counts_correct=None, counts_total=None)

  @classmethod
  def from_model_output(cls, scores: jnp.array, label: jnp.array,
                        **_) -> clu_metrics.Metric:
    num_classes = label.shape[-1]
    if scores.shape[-1] != num_classes:
      raise ValueError(
          "Expected the last dims of `scores` and `label` to be the same.")
    if scores.ndim not in [2, 3]:
      raise ValueError("Expecting the scores to be in one of the following"
                       "formats: [n_devices, batch_size, n_classes] or"
                       "[batch_size, n_classes]. Current shape is"
                       f"{scores.shape}")
    if label.ndim not in [2, 3]:
      raise ValueError("Expecting the label to be in one of the following"
                       "formats: [n_devices, batch_size, n_classes] or"
                       "[batch_size, n_classes]. Current shape is"
                       f"{label.shape}")
    is_correct = (scores.argmax(axis=-1) == label.argmax(axis=-1)
                 )  # [*, batch_size].
    is_correct_flat = jnp.reshape(is_correct, [1, -1])
    label_flat = jnp.reshape(label, [-1, num_classes])
    # [1, *] x [*, num_classes] where * = batch_size or n_devices * batch_size.
    per_class_correct_counts = jnp.squeeze(
        jnp.matmul(is_correct_flat, label_flat))  # [num_classes].
    counts = jnp.sum(label_flat, axis=0).astype(jnp.float32)  # [num classes].
    return cls(counts_correct=per_class_correct_counts, counts_total=counts)

  def merge(self, other):
    if self.counts_correct is None:
      assert self.counts_total is None
      counts_correct = other.counts_correct
      counts_total = other.counts_total
    elif other.counts_correct is None:
      assert other.counts_total is None
      counts_correct = self.counts_correct
      counts_total = self.counts_total
    else:
      num_classes = self.counts_correct.shape[-1]

      # [num_devices, new bs, num_classes] or [new bs, num_classes] where new bs
      # is the combined batch size.
      counts_correct = jnp.concatenate(
          (jnp.reshape(self.counts_correct, [-1, num_classes]),
           jnp.reshape(other.counts_correct, [-1, num_classes])),
          axis=-2)
      counts_total = jnp.concatenate(
          (jnp.reshape(self.counts_total, [-1, num_classes]),
           jnp.reshape(other.counts_total, [-1, num_classes])),
          axis=-2)
      # [num_devices, num_classes] or [num_classes].
      counts_correct = jnp.sum(counts_correct, axis=-2)
      counts_total = jnp.sum(counts_total, axis=-2)
    return type(self)(
        counts_correct=counts_correct,
        counts_total=counts_total,
    )

  def compute(self) -> Any:
    """Compute cmap only using classes that have > sample_threshold samples."""
    per_class_acc = self.counts_correct / self.counts_total
    mca = jnp.mean(per_class_acc)
    return mca, per_class_acc


def make_mca_metric():
  """Create an empty MAC metric."""
  return MCA.empty()


def update_mca_metric(mca_metric, model_outputs, batch):
  """Update a mac_metric from model_outputs and a batch."""
  label = batch["label"].astype(np.int32)
  new_metric = MCA.from_model_output(model_outputs.label, label)
  mca_metric = mca_metric.merge(new_metric)
  return mca_metric
