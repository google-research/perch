# coding=utf-8
# Copyright 2023 The Perch Authors.
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

"""Weakly supervised models and losses."""
import math

from chirp.models import efficientnet
from chirp.models import frontend
from flax import linen as nn
from jax import lax
from jax import numpy as jnp
import optax


def group_reduce(
    elements,
    sentinels,
    reduction_operator=jnp.maximum,
    initializer=lambda x: jnp.full_like(x, jnp.finfo(x.dtype).min),
):
  """Apply a reduction operator over groups.

  Args:
    elements: An array whose leading dimension contains elements of the groups.
    sentinels: A boolean vector which signifies whether an element is the last
      one in its group.
    reduction_operator: The reduction operator to apply to the elements within a
      group. Must take two elements and return a single one.
    initializer: A function which generates the initial values to use for the
      reduction operator.

  Returns:
    An array of the same size as elements. The last element of each group (as
    identified by the sentinels) is replaced with the reduced value. All the
    other elements are replaced with zeros.
  """

  def step(state, x):
    element, sentinel = x
    reduction = reduction_operator(state, element)
    state = lax.select(sentinel, initializer(state), reduction)
    out = lax.select(sentinel, reduction, jnp.zeros_like(element))
    return state, out

  _, reduced = lax.scan(step, initializer(elements[0]), (elements, sentinels))
  return reduced


def weakly_supervised_sigmoid_binary_cross_entropy(logits, labels, sentinels):
  """Sigmoid binary cross entropy for multiple instance learning.

  This function takes the maximum logits over all examples within a bag. It then
  scores these against the labels using sigmoid binary cross entropy. It returns
  the sum (across bags) of the mean (across classes) entropy. Note that it
  returns the sum because in a multi-device setting different devices might have
  processed a different number of bags. Hence the normalization should happen
  only after the loss has been aggregated across devices.

  Args:
    logits: The logits of all examples across bags.
    labels: The labels for each example. Note that only the labels corresponding
      to the last example in each bag are used. The rest are ignored.
    sentinels: A binary mask which signifies whether an example is the last
      example in the bag.

  Returns:
    The sum of the mean binary cross entropy for each bag.
  """
  group_logits = group_reduce(logits, sentinels)
  losses = optax.sigmoid_binary_cross_entropy(group_logits, labels)
  losses = losses * sentinels[:, None]
  return jnp.sum(jnp.mean(losses, axis=-1))


class InstanceModel(nn.Module):
  """An instance-level model for weakly supervised learning.

  This just creates a spectrogram for each window and runs it through an
  EfficientNet model for embedding.

  Attributes:
    num_classes: The number of classes in the weakly supervised learning
      problem.
    stride: The stride used by the frontend.
    sample_rate: The sample rate of the audio (used by the frontend).
    freq_range: The frequency range used for the spectrograms.
  """

  num_classes: int
  stride: int = 32_000 // 100
  sample_rate: int = 32_000
  freq_range: tuple[int, int] = (60, 16_000)

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      use_running_average: bool | None = None,
      project: bool = True,
  ) -> jnp.ndarray:
    spectrograms = self.frontend(inputs)
    self.sow('intermediates', 'spectrograms', spectrograms)
    embeddings = self.encode(spectrograms, train, use_running_average)
    self.sow('intermediates', 'embeddings', embeddings)
    return nn.Dense(self.num_classes)(embeddings)

  def frontend(self, inputs: jnp.ndarray) -> jnp.ndarray:
    kernel_size = 2 * self.stride
    nfft = 2 ** math.ceil(math.log2(kernel_size))
    # TODO(bartvm): Confirm whether log-scaling works well here
    return frontend.MelSpectrogram(
        features=128,
        stride=self.stride,
        kernel_size=kernel_size,
        sample_rate=self.sample_rate,
        freq_range=self.freq_range,
        power=1.0,
        scaling_config=frontend.LogScalingConfig(floor=1e-5),
        nfft=nfft,
    )(inputs)[..., None]

  def encode(
      self, inputs: jnp.ndarray, train: bool, use_running_average: bool
  ) -> jnp.ndarray:
    return efficientnet.EfficientNet(efficientnet.EfficientNetModel.B0)(
        inputs, train=train, use_running_average=use_running_average
    )
