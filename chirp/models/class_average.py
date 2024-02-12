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

"""A CLU metric that averages values separately for each class."""
from typing import Any

from clu import metrics
import flax
from jax import numpy as jnp


@flax.struct.dataclass
class ClassAverage(metrics.Metric):
  """A metric that calculates the average over each class.

  This metric assumes that it's given a score for each example in the batch
  along with a mapping from batches to (potentially multiple) classes in the
  form a multi-hot encoding.
  """

  total: jnp.ndarray
  count: jnp.ndarray

  @classmethod
  def empty(cls):
    return cls(total=jnp.zeros((1,), float), count=jnp.zeros((1,), int))

  @classmethod
  def from_model_output(
      cls, values: tuple[jnp.ndarray, jnp.ndarray], **_
  ) -> metrics.Metric:
    return cls(total=values[0] @ values[1], count=jnp.sum(values[1], axis=0))

  def merge(self, other: "ClassAverage") -> "ClassAverage":
    return type(self)(
        total=self.total + other.total,
        count=self.count + other.count,
    )

  def compute(self) -> Any:
    # Avoid introducing NaNs due to classes without positive labels
    class_means = jnp.where(self.count > 0, self.total / self.count, 0.0)
    return jnp.sum(class_means) / jnp.sum(self.count > 0)
