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

"""Continuous wavelet transforms.

This module contains filters (Gabor, sinc) and wavelets (Morlet, Morse) that can
be used in a continuous wavelet transform.

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
from typing import Callable

import chirp.signal
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp


class Domain(enum.Enum):
  TIME = "time"
  FREQUENCY = "frequency"


class Normalization(enum.Enum):
  L1 = "l1"
  L2 = "l2"


def gabor_filter(
    sigma: float, domain: Domain, normalization: Normalization
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """A one-dimensional Gabor filter.

  The Gabor filter is a complex sinusoid modulated by a Gaussian. Its frequency
  response is a Gaussian with mean sigma / 2π and standard deviation 1 / 2π[^1].

  For small values of sigma this filter has a non-zero response to non-positive
  frequencies. This means that it is not a wavelet (it fails the admissibility
  condition)[^2] and it is not analytic[^3].

  The zero-mean, shifted version of this filter is often called the (complex)
  Morlet wavelet, Gabor wavelet, or Gabor kernel. The Gabor filter is sometimes
  also referred to as the Gabor function[^4].

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
    A function which calculates the filter over the time or frequency domain.
  """
  if domain is Domain.TIME:
    if normalization is Normalization.L1:
      norm = 1 / jnp.sqrt(2 * jnp.pi)
    elif normalization is Normalization.L2:
      norm = jnp.pi ** (-1 / 4)

    def _gabor_filter(t: jnp.ndarray) -> jnp.ndarray:
      sinusoids = jnp.exp(1j * t * sigma)
      gaussian = jnp.exp(-1 / 2 * t**2)
      return norm * gaussian * sinusoids

  elif domain is Domain.FREQUENCY:
    if normalization is Normalization.L1:
      norm = 1.0
    elif normalization is Normalization.L2:
      norm = jnp.pi ** (1 / 4) * jnp.sqrt(2)

    def _gabor_filter(f: jnp.ndarray) -> jnp.ndarray:
      gaussian = jnp.exp(-1 / 2 * (sigma - f * 2 * jnp.pi) ** 2)
      return norm * gaussian

  return _gabor_filter


def sinc_filter(
    sigma: float, domain: Domain, normalization: Normalization
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """A sinc filter.

  Rather than being parameterized by its upper and lower frequency, this sinc
  filter is parametrized by its central frequency. The width of the filter can
  then be set by the scaling factor (i.e., scaling the inputs).

  The sinc filter is not differentiable in the frequency domain.

  The L1 norm of the sinc filter diverges to infinity[^1], so L1 normalization
  is not supported.

  [^1]: Borwein, David, Jonathan M. Borwein, and Isaac E. Leonard. "L p norms
    and the sinc function." The American Mathematical Monthly 117.6 (2010):
    528-539.

  Args:
    sigma: The central frequency of the function.
    domain: The domain.
    normalization: What normalization to use.

  Returns:
    A function which calculates the filter over the time or frequency domain.

  Raises:
    ValueError: If L1 normalization is requested.
  """
  if normalization is Normalization.L1:
    raise ValueError("sinc filter does not support L1 normalization")
  if domain is Domain.TIME:

    def _sinc_filter(t: jnp.ndarray) -> jnp.ndarray:
      shift = jnp.exp(2j * jnp.pi * t * sigma)
      # NOTE: Normalized sinc function
      return shift * jnp.sinc(t)

  elif domain is Domain.FREQUENCY:

    def _sinc_filter(f: jnp.ndarray) -> jnp.ndarray:
      return jnp.where(jnp.abs(f - sigma) < 1 / 2, 1.0, 0.0)

  return _sinc_filter


def morlet_wavelet(
    sigma: float, domain: Domain, normalization: Normalization
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """A Morlet wavelet.

  This wavelet is a sinusoid modulated by a Gaussian which is shifted down in
  order to have zero mean (admissibility condition). It has a non-zero response
  to negative frequencies, so it is not analytic.

  For large values of sigma this wavelet is approximately equal to a Gabor
  filter. See `gabor_filter` for details regarding naming.

  The peak frequency of this wavelet is the solution `wc` to the equation
  wc * 2π = sigma / (1 - exp(-sigma * wc * 2π)). This can be found using fixed
  point iteration.

  Args:
    sigma: The parameter which allows the wavelet to trade-off between time and
      frequency resolution.
    domain: The domain.
    normalization: What normalization to use.

  Returns:
    A function which calculates the filter over the time or frequency domain.
  """
  if normalization is Normalization.L1:
    # TODO(bartvm): Does an expression exist for this?
    raise NotImplementedError

  # Follows notation from, e.g., https://en.wikipedia.org/wiki/Morlet_wavelet
  kappa = jnp.exp(-1 / 2 * sigma**2)
  c = (1 + jnp.exp(-(sigma**2)) - 2 * jnp.exp(-3 / 4 * sigma**2)) ** (
      -1 / 2
  )

  if domain is Domain.TIME:

    def _morlet_wavelet(t: jnp.ndarray) -> jnp.ndarray:
      return (
          c
          * jnp.pi ** (-1 / 4)
          * jnp.exp(-1 / 2 * t**2)
          * (jnp.exp(1j * sigma * t) - kappa)
      )

  elif domain is Domain.FREQUENCY:

    def _morlet_wavelet(f: jnp.ndarray) -> jnp.ndarray:
      f = jnp.pi * 2 * f
      return (
          c
          * jnp.pi ** (1 / 4)
          * jnp.sqrt(2)
          * (
              jnp.exp(-1 / 2 * (sigma - f) ** 2)
              - kappa * jnp.exp(-1 / 2 * f**2)
          )
      )

  return _morlet_wavelet


def morse_wavelet(
    gamma: float, beta: float, domain: Domain, normalization: Normalization
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """A Morse wavelet.

  For a general overview of Morse wavelets see Lilly and Olhede[^1][^2]. For
  the mathematical details of higher-order wavelets, see Olhede and Walden[^3].
  This code follows the notation in Lilly and Olhede.

  This wavelet is analytic (i.e., it has zero response for negative
  frequencies). It has no analytic expression in the time-domain.

  The peak frequency is equal to (beta / gamma)^(1 / gamma) and the width
  (duration) scales as sqrt(beta * gamma).

  [^1]: Lilly, Jonathan M., and Sofia C. Olhede. "Generalized Morse wavelets as
      a superfamily of analytic wavelets." IEEE Transactions on Signal
      Processing 60.11 (2012): 6036-6041.
  [^2]: Lilly, Jonathan M., and Sofia C. Olhede. "Higher-order properties of
      analytic wavelets." IEEE Transactions on Signal Processing 57.1 (2008):
      146-160.
  [^3]: Olhede, Sofia C., and Andrew T. Walden. "Generalized morse wavelets."
      IEEE Transactions on Signal Processing 50.11 (2002): 2661-2670.

  Args:
    gamma: A parameter which controls the high-frequency decay. A common choice
      is 3, in which case it defines the family of Airy wavelets (which are
      similar to the commonly used Morlet and Gabor wavelets). See figure 1 in
      Lilly and Olhede (2012) for the relationship between gamma and other
      wavelet families. Gamma must be positive.
    beta: A parameter which controls the behavior near the zero frequency. When
      gamma is equal to 3, increasing this has a similar effect as increasing
      the parameter of a Morlet wavelet. Beta must be positive.
    domain: The domain.
    normalization: What normalization to use.

  Returns:
    A function which calculates the wavelet over the frequency domain.
  """
  if domain is not Domain.FREQUENCY:
    raise ValueError(
        "Morse wavelets have no analytic expression in the time domain"
    )

  r = (2 * beta + 1) / gamma

  # NOTE: Computations in log-space for numerical stability
  if normalization is Normalization.L2:
    log_norm = (jnp.log(gamma) + r * jnp.log(2) - jsp.special.gammaln(r)) / 2
  elif normalization is Normalization.L1:
    log_norm = jnp.log(gamma) - jsp.special.gammaln((1 + beta) / gamma)

  def _morse_wavelet(f: jnp.ndarray) -> jnp.ndarray:
    f_nonneg = f >= 0
    f *= f_nonneg
    return jnp.exp(log_norm + beta * jnp.log(f) - f**gamma) * f_nonneg

  return _morse_wavelet


def melspec_params(
    num_mel_bins: int,
    sample_rate: float,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
) -> jnp.ndarray:
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
  range_ = map(chirp.signal.hertz_to_mel, (lower_edge_hertz, upper_edge_hertz))  # pytype: disable=wrong-arg-types  # jax-ndarray
  bands = chirp.signal.mel_to_hertz(jnp.linspace(*range_, num_mel_bins + 2))

  # Convert from Hertz to normalized frequencies
  bands = bands / sample_rate * jnp.pi * 2

  # Triangle filters with peak 1, but we take the square root so the the slopes
  # reach the half maximum 1/2 at 1/4
  fwhms = ((3 * bands[2:] + bands[1:-1]) - (bands[1:-1] + 3 * bands[:-2])) / 4
  # To convert from FWHM to standard deviation for a Gaussian
  coeff = 2 * jnp.sqrt(2 * jnp.log(2))
  inv_fwhms = coeff / fwhms

  return bands[1:-1], inv_fwhms  # pytype: disable=bad-return-type  # jax-ndarray


def convolve_filter(
    filter_: Callable[[jnp.ndarray], jnp.ndarray],
    signal: jnp.ndarray,
    scale_factors: jnp.ndarray,
    normalization: Normalization,
    window_size_frames: int,
    stride: tuple[int, ...] = (1,),
    padding: str = "SAME",
) -> jnp.ndarray:
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
  if normalization is Normalization.L1:
    norm = 1 / scale_factors
  elif normalization is Normalization.L2:
    norm = 1 / jnp.sqrt(scale_factors)
  sampled_filters = norm * jnp.conj(filter_(ts))

  # We assume a single input channel
  sampled_filters = sampled_filters[:, jnp.newaxis]
  dn = lax.conv_dimension_numbers(
      signal.shape, sampled_filters.shape, ("NWC", "WIO", "NWC")
  )
  # TODO(bartvm): Not all platforms (e.g., TF Lite) support complex inputs for
  # convolutions. Can be addressed by convolving with the real/imaginary parts
  # separately in the future if needed.
  # TODO(bartvm): Converting signal to complex because JAX wants the input and
  # filters to be the same type, but this results in 33% more multiplications
  # than necessary, so this is probably not the fastest option.
  signal = signal.astype(jnp.complex64)
  filtered_signal = lax.conv_general_dilated(
      signal, sampled_filters, stride, padding, (1,), (1,), dn
  )
  return filtered_signal


def multiply_filter(
    filter_: Callable[[jnp.ndarray], jnp.ndarray],
    signal: jnp.ndarray,
    scale_factors: jnp.ndarray,
    normalization: Normalization,
) -> jnp.ndarray:
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
      jnp.fft.fft(signal, axis=-2) * filter_(fs), axis=-2
  )
  if normalization is Normalization.L1:
    norm = 1
  elif normalization is Normalization.L2:
    norm = jnp.sqrt(scale_factors)
  return norm * filtered_signal
