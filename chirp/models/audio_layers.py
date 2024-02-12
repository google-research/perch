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

"""Flax layers for audio processing."""
import functools

from chirp import audio_utils
from flax import linen as nn
from jax import lax
from jax import numpy as jnp
from jax import random


class RandomLowPassFilter(nn.Module):
  """A random low-pass filter in the frequency-domain.

  Attributes:
    rate: The rate at which random low-pass filters are applied.
    deterministic: If true, no low-pass filters are applied.
  """

  rate: float
  deterministic: bool | None = None

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, deterministic: bool | None = None
  ) -> jnp.ndarray:
    """Applies a random low-pass filter to a mel-spectrogram.

    Args:
      inputs: A (batch) of mel-spectrograms, assumed to have frequencies on the
        last axis.
      deterministic: If true, passes the input as is.

    Returns:
      Aspectrogram with the same size as the input, possibly with a random
      low-pass filter applied.
    """
    deterministic = nn.merge_param(
        'deterministic', self.deterministic, deterministic
    )
    if self.rate == 0.0 or deterministic:
      return inputs
    rng = self.make_rng('low_pass')
    rate_key, low_pass_key = random.split(rng)
    x = lax.cond(
        random.uniform(rate_key) < self.rate,
        functools.partial(audio_utils.random_low_pass_filter, low_pass_key),
        lambda x: x,
        inputs,
    )

    return x
