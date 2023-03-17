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

"""Custom metrics for SFDA."""
from chirp.projects.sfda import losses
from clu import metrics as clu_metrics
import flax
import jax.numpy as jnp


@flax.struct.dataclass
class Accuracy(clu_metrics.Average):
  """Computes the accuracy from model outputs `probabilities` and `labels`.

  `label` is expected to be of dtype=int32 and to have 0 <= ndim <= 2, and
  `probabilities` is expected to have ndim = labels.ndim + 1.

  See also documentation of `Metric`.
  """

  @classmethod
  def from_model_output(
      cls, probabilities: jnp.ndarray, label: jnp.ndarray, **kwargs
  ) -> clu_metrics.Metric:
    return super().from_model_output(
        values=(probabilities.argmax(axis=-1) == label.argmax(axis=-1)).astype(
            jnp.float32
        ),
        **kwargs
    )


@flax.struct.dataclass
class MarginalEntropy(clu_metrics.Metric):
  """Computes the marginal entropy of a model's output distribution.

  Marginal entropy is a useful metric to track. To compute it, one requires
  computing the marginal label distribution of the model, which requires access
  to all of the model's outputs on batch of interest. That makes it a
  non-`averageable` metric, which is why we dedicate a separate metric for it.
  """

  probability_sum: jnp.ndarray
  n_samples: int
  multi_label: bool
  label_mask: jnp.ndarray | None

  @classmethod
  def from_model_output(
      cls, probabilities: jnp.ndarray, multi_label: bool, label_mask: jnp.ndarray,
      **_
  ) -> "MarginalEntropy":
    return cls(
        probability_sum=probabilities.sum(axis=0),
        n_samples=probabilities.shape[0],
        multi_label=multi_label,
        label_mask=label_mask
    )

  def merge(self, other: "MarginalEntropy") -> "MarginalEntropy":
    return type(self)(
        probability_sum=self.probability_sum + other.probability_sum,
        n_samples=self.n_samples + other.n_samples,
        multi_label=other.multi_label,
        label_mask=other.label_mask
    )

  def compute(self):
    proba_marginal = self.probability_sum * (1 / self.n_samples)
    reference_mask = None if self.label_mask is None else self.label_mask[0]
    return losses.label_ent(proba_marginal, reference_mask)

  @classmethod
  def empty(cls) -> "MarginalEntropy":
    return cls(probability_sum=0.0, n_samples=0, multi_label=False,
               label_mask=None)


@flax.struct.dataclass
class MarginalBinaryEntropy(clu_metrics.Metric):
  """A version of MarginalEntropy, for binary entropies.

  TODO(mboudiaf) Merge this metric with MarginalEntropy using jax.lax.cond on
  multi_label.
  """

  probability_sum: jnp.ndarray
  n_samples: int
  label_mask: jnp.ndarray
  multi_label: bool

  @classmethod
  def from_model_output(
      cls,
      label_mask: jnp.ndarray,
      probabilities: jnp.ndarray,
      multi_label: bool,
      **_
  ) -> "MarginalBinaryEntropy":
    # TODO(mboudiaf). Verify here that label_mask is the same across samples.
    # Problem is to make assert inside jitted function. Right now, this is done
    # before each iteration, but ideally should be here.

    return cls(
        probability_sum=probabilities.sum(axis=0),
        label_mask=label_mask[0],
        n_samples=probabilities.shape[0],
        multi_label=multi_label,
    )

  def merge(self, other: "MarginalBinaryEntropy") -> "MarginalBinaryEntropy":
    return type(self)(
        probability_sum=self.probability_sum + other.probability_sum,
        n_samples=self.n_samples + other.n_samples,
        label_mask=other.label_mask,
        multi_label=other.multi_label,
    )

  def compute(self):
    proba_marginal = self.probability_sum * (1 / self.n_samples)
    return losses.label_binary_ent(
        probabilities=proba_marginal, label_mask=self.label_mask
    )

  @classmethod
  def empty(cls) -> "MarginalBinaryEntropy":
    return cls(
        probability_sum=0.0, n_samples=0, label_mask=0.0, multi_label=False
    )
