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

"""Learned pooling module."""
from typing import Callable

from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp


def gaussian(M: int, std: float, sym: bool = True) -> jnp.ndarray:  # pylint: disable=invalid-name
  """Returns a Gaussian window.

  Port of scipy.signal.windows.gaussian.

  Args:
    M: Number of points in the output window.
    std: The standard deviation, sigma.
    sym: Must be `True` (present for compatibility with SciPy' signature).

  Returns:
    The window, with the maximum value normalized to 1 (though the value 1 does
    not appear if M is even).
  """
  if not sym:
    raise ValueError("Periodic windows not supported")

  n = jnp.arange(0, M) - (M - 1.0) / 2.0
  sig2 = 2 * std * std
  w = jnp.exp(-(n**2) / sig2)
  return w


def gaussian_init(
    key: jnp.ndarray, num_channels: int, window_size: int, std: float = 0.4
) -> jnp.ndarray:
  """Initializes Gaussian windows.

  Args:
    key: RNG, unused.
    num_channels: The number of windows to calculate.
    window_size: The number of steps in the window (which is assumed to range
      from -1 to 1).
    std: The standard deviation of the Gaussian.

  Returns:
    A one-tuple containing an array with `num_channels` entries. These represent
    the standard deviation scaled by the window size.
  """
  del key
  return (std * 0.5 * (window_size - 1) * jnp.ones((num_channels,)),)  # pytype: disable=bad-return-type  # jax-ndarray


class WindowPool(nn.Module):
  """Pools using a window function.

  Note that is not a pooling function in the traditional sense, i.e., it does
  not use a reduction operator applied to the elements in each window. Instead,
  a weighted average is taken over the window. If the weighting is given by a
  parametrized window, e.g., a Gaussian, then these parameters are learned. This
  allows the model to interpolate between subsampling (a Gaussian with zero
  variance) and average pooling (a Gaussian with infinite variance).

  When using a Gaussian window, there are a few differences with the
  implementation in LEAF[^1]. Firstly, this module by default scales the weights
  to sum to unity. This ensure that the energy of the output signal is the same
  as the input. Secondly, this module does not perform clipping on the window
  parameters. This is expected to be done during optimization.

  [^1]: https://github.com/google-research/leaf-audio

  Attributes:
    window: The window function to use. Should follow the conventions of the
      `scipy.signal.windows` functions.
    window_size: The size of the pooling window.
    window_init: Initializer of the window parameters. It should take as an
      argument an RNG key, the number of filters, and the width of the window,
      and return a tuple of parameters. Each parameter should have the number of
      filters as its first axis.
    normalize_window: Whether or not to normalize the window to sum to 1.
    stride: The stride to use.
    padding: Padding to use.
  """

  window: Callable[..., jnp.ndarray]
  window_size: int
  window_init: Callable[[jnp.ndarray, int, int], jnp.ndarray]
  normalize_window: bool = True
  stride: int = 1
  padding: str = "SAME"

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies the pooling.

    Args:
      inputs: The input array must be of shape `(batch, time, channels)`. Each
        channel will have its own window applied. In the case of a parametrized
        window, each channel will have its own parameters.

    Returns:
      The pooled outputs of shape (batch, time, channels).
    """
    num_channels = inputs.shape[-1]
    window_params = self.param(
        "window_params", self.window_init, num_channels, self.window_size
    )
    window_values = jax.vmap(
        self.window, in_axes=(None,) + (0,) * len(window_params)
    )(self.window_size, *window_params)
    if self.normalize_window:
      window_values /= jnp.sum(window_values, axis=1, keepdims=True)
    window_values = window_values.T[:, jnp.newaxis]
    dn = lax.conv_dimension_numbers(
        inputs.shape, window_values.shape, ("NWC", "WIO", "NWC")
    )

    return lax.conv_general_dilated(
        inputs,
        window_values,
        (self.stride,),
        self.padding,
        dimension_numbers=dn,
        feature_group_count=num_channels,
    )
