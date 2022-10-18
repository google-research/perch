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
  def from_model_output(cls, *, probabilities: jnp.array, label: jnp.array,
                        **kwargs) -> clu_metrics.Metric:
    return super().from_model_output(
        values=(probabilities.argmax(axis=-1) == label.argmax(axis=-1)).astype(
            jnp.float32),
        **kwargs)
