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

"""Learnable filter banks.

This defines a module which learns a filter bank. The filters can be
parametrized functions (Gabor, sinc), wavelets (Morlet, Morse), or entirely
learned (1D convolutional filters).

Filtering can be done through convolution in the time-domain or by
multiplication in the frequency domain. Note that some filters are
non-differentiable in the frequency domain (sinc filters) whereas others don't
have an analytic representation in the time domain (Morse wavelets). Hence we
use both approaches.

Filters and wavelets can be normalized in different ways. Let φ(t) be the
time-domain filter and Φ(f) the frequency-domain filter. Different normalization
options are:

  * ∫|φ(t)| = 1 (L1 normalized, bandpass normalized)
  * ∫|φ(t)|² = ∫|Φ(t)|² = 1 (L2 normalized, energy normalized)
  * max |Φ(f)| = 1 (peak frequency response)

Note that max |Φ(f)| ≤ ∫|φ(t)|.

There are a variety of Fourier transform conventions. This code follows the
one from NumPy where the ordinary frequency is used: f̂(ω) = ∫f(t)exp(-i2πωt)dt.

For all parametrized filters we follow the approach of the continuous wavelet
transform (CWT) and learn a scale for each filter: φ(t/s) for s > 0. To keep the
function norm unchanged across scalings, we have to scale the values by 1/s (for
L1 normalization, time domain only) or 1/√s (for L2 normalization).

For reference implementations, see the Python package `ssqueezepy` and the
MATLAB package jLab.
"""
import enum
from typing import Callable, Tuple

import chirp.signal
from jax import lax
from jax import numpy as jnp


class Domain(enum.Enum):
  TIME = "time"
  FREQUENCY = "frequency"


class Normalization(enum.Enum):
  L1 = "l1"
  L2 = "l2"


def gabor_filter(
    sigma: float, domain: Domain,
    normalization: Normalization) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """A one-dimensional Gabor filter.

  The Gabor filter is a complex sinusoid modulated by a Gaussian. Its frequency
  response is a Gaussian with mean sigma / 2π and standard deviation 1 / 2π[^1].

  For small values of sigma this filter has a non-zero response to non-positive
  frequencies. This means that it is not a wavelet (it fails the admissibility
  condition)[^2] and it is not analytic[^3].

  The zero-mean, shifted version of this filter is often called the Morlet
  wavelet, Gabor wavelet, or Gabor kernel. The Gabor filter is sometimes also
  referred to as the Gabor function[^4].

  To match `leaf_audio.impulse_response.gabor_impulse_response`: Use L1
  normalization. Set sigma to the product of η and σ and then use σ as the
  scaling factor.

  [^1]: Movellan, Javier R. "Tutorial on Gabor filters." (2002).
  [^2]: Valens, Clemens. "A really friendly guide to wavelets." (1999).
  [^3]: Lilly, Jonathan M., and Sofia C. Olhede. "On the analytic wavelet
    transform." IEEE transactions on information theory 56.8 (2010): 4135-4156.
  [^4]: Bernardino, Alexandre, and José Santos-Victor. "A real-time gabor primal
    sketch for visual attention." Iberian Conference on Pattern Recognition and
    Image Analysis. Springer, Berlin, Heidelberg, 2005.

  Args:
    sigma: The parameter of the function.
    domain: The domain.
    normalization: What normalization to use.

  Returns:
    A function which calculates the filter over the time domain.
  """
  if domain == Domain.TIME:
    if normalization == Normalization.L1:
      norm = 1 / jnp.sqrt(2 * jnp.pi)
    elif normalization == Normalization.L2:
      norm = jnp.pi**(-1 / 4)

    def _gabor_filter(t: jnp.ndarray) -> jnp.ndarray:
      sinusoids = jnp.exp(1j * t * sigma)
      gaussian = jnp.exp(-1 / 2 * t**2)
      return norm * gaussian * sinusoids
  elif domain == Domain.FREQUENCY:
    if normalization == Normalization.L1:
      norm = 1.
    elif normalization == Normalization.L2:
      norm = jnp.pi**(1 / 4) * jnp.sqrt(2)

    def _gabor_filter(f: jnp.ndarray) -> jnp.ndarray:
      gaussian = jnp.exp(-1 / 2 * (sigma - f * 2 * jnp.pi)**2)
      return norm * gaussian

  return _gabor_filter


def melspec_params(num_mel_bins: int, sample_rate: float,
                   lower_edge_hertz: float,
                   upper_edge_hertz: float) -> jnp.ndarray:
  """Gets the peak frequencies and bandwidths of a standard mel-filterbank.

  This assumes a Gaussian frequency response (i.e., Gabor filters) and matches
  the full width at half maximum (FWHM) of the square root of the triangle
  filter to the FWHM of this Gaussian.

  Args:
    num_mel_bins: The number of mel bandpass filters.
    sample_rate: The sampling rate of the signal. Used to calculate the Nyquist
      frequency, which determines the upper bound on frequencies in the signal.
    lower_edge_hertz: The lowest frequency to generate filters for.
    upper_edge_hertz: The highest frequency to generate filters for.

  Returns:
    The central frequencies and the inverse bandwidth (normalized).
  """
  # The melspec triangle filters are equally spaced in the mel-scale
  range_ = map(chirp.signal.hertz_to_mel, (lower_edge_hertz, upper_edge_hertz))
  bands = chirp.signal.mel_to_hertz(jnp.linspace(*range_, num_mel_bins + 2))

  # Convert from Hertz to normalized frequencies
  bands = bands / sample_rate * jnp.pi * 2

  # Triangle filters with peak 1, but we take the square root so the the slopes
  # reach the half maximum 1/2 at 1/4
  fwhms = ((3 * bands[2:] + bands[1:-1]) - (bands[1:-1] + 3 * bands[:-2])) / 4
  # To convert from FWHM to standard deviation for a Gaussian
  coeff = 2 * jnp.sqrt(2 * jnp.log(2))
  inv_fwhms = coeff / fwhms

  return bands[1:-1], inv_fwhms


def convolve_filter(filter_: Callable[[jnp.ndarray], jnp.ndarray],
                    signal: jnp.ndarray,
                    scale_factors: jnp.ndarray,
                    normalization: Normalization,
                    window_size_frames: int,
                    stride: Tuple[int, ...] = (1,),
                    padding: str = "SAME") -> jnp.ndarray:
  """Convolves a given set of filters with a signal in the time domain.

  Note that this takes the conjugate of the filter in order to match the usual
  conventions of continuous wavelet transforms.

  Args:
    filter_: A time-domain filter which takes radians as inputs.
    signal: A batch of signals (assumed to be in the format `NWC`).
    scale_factors: Each filter has a scale associated with it.
    normalization: The normalization to use.
    window_size_frames: The width of the window to use, in frames.
    stride: The stride to use for the convolution.
    padding: The padding to use for the convolution.

  Returns:
    The signals filtered with the given filters.
  """
  ts = jnp.arange(-(window_size_frames // 2), (window_size_frames + 1) // 2)
  ts = ts[:, jnp.newaxis] / scale_factors
  if normalization == Normalization.L1:
    norm = 1 / scale_factors
  elif normalization == Normalization.L2:
    norm = 1 / jnp.sqrt(scale_factors)
  sampled_filters = norm * jnp.conj(filter_(ts))

  # We assume a single input channel
  sampled_filters = sampled_filters[:, jnp.newaxis]
  dn = lax.conv_dimension_numbers(signal.shape, sampled_filters.shape,
                                  ("NWC", "WIO", "NWC"))
  # TODO(bartvm): Not all platforms (e.g., TF Lite) support complex inputs for
  # convolutions. Can be addressed by convolving with the real/imaginary parts
  # separately in the future if needed.
  # TODO(bartvm): Converting signal to complex because JAX wants the input and
  # filters to be the same type, but this results in 33% more multiplications
  # than necessary, so this is probably not the fastest option.
  signal = signal.astype(jnp.complex64)
  filtered_signal = lax.conv_general_dilated(signal, sampled_filters, stride,
                                             padding, (1,), (1,), dn)
  return filtered_signal


def multiply_filter(filter_: Callable[[jnp.ndarray], jnp.ndarray],
                    signal: jnp.ndarray, scale_factors: jnp.ndarray,
                    normalization: Normalization) -> jnp.ndarray:
  """Applies a filter to a signal in the frequency domain.

  This takes the DFT of the given signals and applies the given filter.

  Args:
    filter_: The filter in the frequency domain to apply.
    signal: A batch of signals, assumed to have time and channels as the last
      two axes.
    scale_factors: The scale factors to apply to each kernel.
    normalization: The normalization to use.

  Returns:
    The result of applying the filter to the signal.
  """
  *_, num_frames, _ = signal.shape
  fs = jnp.fft.fftfreq(num_frames)
  fs = fs[:, jnp.newaxis] * scale_factors
  # TODO(bartvm): TF Lite might not support IFFT as a built-in operation, but
  # IFFT is just an FFT with the sign of the inputs changed so easy to adapt to.
  # TODO(bartvm): Note that the signal is real-valued, so using FFT might do
  # unnecessary computation. Might be faster to use RFFT and then take the
  # complex conjugates manually.
  filtered_signal = jnp.fft.ifft(
      jnp.fft.fft(signal, axis=-2) * filter_(fs), axis=-2)
  if normalization == Normalization.L1:
    norm = 1
  elif normalization == Normalization.L2:
    norm = jnp.sqrt(scale_factors)
  return norm * filtered_signal
