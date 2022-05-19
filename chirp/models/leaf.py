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

"""LEAF frontend."""
from typing import Sequence

import chirp.signal
from jax import lax
from jax import numpy as jnp


def gabor_impulse_response(t: jnp.ndarray, center: jnp.ndarray,
                           fwhm: jnp.ndarray) -> jnp.ndarray:
  """Computes a set of Gabor impulse responses.

  Args:
    t: The times at which to calculate the impulse response.
    center: The center frequencies of the kernels.
    fwhm: The bandwidth of the filter in terms of its full width at half maximum
      (FWHM). Must be the same size as `center`.

  Returns:
    An array `(filters, time)`.
  """
  denominator = 1.0 / (jnp.sqrt(2.0 * jnp.pi) * fwhm)
  gaussian = jnp.exp(jnp.tensordot(1.0 / (2. * fwhm**2), -t**2, axes=0))
  center_frequency_complex = center.astype(jnp.complex64)
  t_complex = t.astype(jnp.complex64)
  sinusoid = jnp.exp(1j *
                     jnp.tensordot(center_frequency_complex, t_complex, axes=0))
  denominator = denominator.astype(jnp.complex64)[:, jnp.newaxis]
  gaussian = gaussian.astype(jnp.complex64)
  return denominator * sinusoid * gaussian


def gabor_filters(kernel: jnp.ndarray, size: int) -> jnp.ndarray:
  """Computes the Gabor filters from its parametrization.

  Args:
    kernel: A 2D array of size `(filters, 2)` which parametrized the Gabor
      kernels in terms of its central frequencies and bandwidths.
    size: The size of the window.

  Returns:
    A 2D array of size `(filters, size)`.
  """
  return gabor_impulse_response(
      jnp.arange(-(size // 2), (size + 1) // 2, dtype=jnp.float32),
      center=kernel[:, 0],
      fwhm=kernel[:, 1])


def gabor_constraint(kernel: jnp.ndarray, kernel_size: int) -> jnp.ndarray:
  """Constrain the parameters of Gabor filters.

  This function constrains the central frequencies to the positive part of the
  frequency range. It also constrains the bandwidths so that the full-width at
  half-maximum falls within the window.

  See section 3.1.2 of the paper.

  Args:
    kernel: A 2D array of size `(filters, 2)` which parametrized the Gabor
      kernels in terms of its central frequencies and bandwidths.
    kernel_size: The size of the window.

  Returns:
    A kernel of the same size, but with the central frequencies constrained.
  """
  mu_lower = 0.
  mu_upper = jnp.pi
  sigma_lower = 4 * jnp.sqrt(2 * jnp.log(2)) / jnp.pi
  sigma_upper = kernel_size * jnp.sqrt(2 * jnp.log(2)) / jnp.pi
  clipped_mu = jnp.clip(kernel[:, 0], mu_lower, mu_upper)
  clipped_sigma = jnp.clip(kernel[:, 1], sigma_lower, sigma_upper)
  return jnp.stack([clipped_mu, clipped_sigma], axis=1)


def gabor_conv1d(inputs: jnp.ndarray,
                 kernel: jnp.ndarray,
                 kernel_size: int,
                 stride: Sequence[int] = (1,),
                 padding: str = 'SAME'):
  """Convolve a signal with a set of Gabor filters.

  The kernel is constrained using `gabor_constraint` before being used.

  Args:
    inputs: Inputs in `NWC` format.
    kernel: The Gabor filters parametrized by a `(filters, 2)` matrix, where the
      first column holds the central frequencies and the second column the
      bandwidth of the filters in terms of their full-width at half-maximum.
    kernel_size: The size of the kernels.
    stride: The stride to use. Defaults to (1,).
    padding: The padding to use, defaults to `SAME`.

  Returns:
    The output of the convolution, `NWC` format.
  """
  constrained_kernel = gabor_constraint(kernel, kernel_size)
  filters = gabor_filters(constrained_kernel, kernel_size)
  real_filters = jnp.real(filters)
  img_filters = jnp.imag(filters)
  stacked_filters = jnp.stack([real_filters, img_filters], axis=1)
  stacked_filters = jnp.reshape(stacked_filters,
                                (2 * kernel.shape[0], kernel_size))
  stacked_filters = jnp.expand_dims(
      jnp.transpose(stacked_filters, axes=(1, 0)), axis=1)

  dn = lax.conv_dimension_numbers(inputs.shape, stacked_filters.shape,
                                  ('NWC', 'WIO', 'NWC'))
  out = lax.conv_general_dilated(inputs, stacked_filters, stride, padding, (1,),
                                 (1,), dn)
  return out


def gabor_init(n_filters: int, n_fft: int, sample_rate: int, min_freq: int,
               max_freq: int) -> jnp.ndarray:
  """Initialize a set of Gabor filters to approximate a mel filterbank.

  Args:
    n_filters: The number of filters.
    n_fft: The number of spectrogram bins.
    sample_rate: The sampling rate.
    min_freq: Minimum frequency.
    max_freq: Maximum frequency.

  Returns:
    A set of Gabor filters with approximately the same central frequencies and
    bandwidths as the filter banks in a mel-spectrogram.
  """
  mel_filters = chirp.signal.linear_to_mel_weight_matrix(
      num_mel_bins=n_filters,
      num_spectrogram_bins=n_fft // 2 + 1,
      sample_rate=sample_rate,
      lower_edge_hertz=min_freq,
      upper_edge_hertz=max_freq)
  mel_filters = jnp.transpose(mel_filters, (1, 0))

  coeff = jnp.sqrt(2. * jnp.log(2.)) * n_fft
  sqrt_filters = jnp.sqrt(mel_filters)
  center_frequencies = jnp.argmax(sqrt_filters, axis=1).astype(jnp.float32)
  peaks = jnp.max(sqrt_filters, axis=1, keepdims=True)
  half_magnitudes = peaks / 2.
  fwhms = jnp.sum((sqrt_filters >= half_magnitudes).astype(jnp.float32), axis=1)
  return jnp.stack(
      [center_frequencies * 2 * jnp.pi / n_fft, coeff / (jnp.pi * fwhms)],
      axis=1)
