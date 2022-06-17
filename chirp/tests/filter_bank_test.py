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

"""Tests for filter_bank."""
from chirp.models import filter_bank
from leaf_audio import convolution
from leaf_audio import initializers
from leaf_audio import melfilters
import numpy as np
import scipy as sp
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized


class FilterBankTest(parameterized.TestCase):

  @parameterized.product(
      beta=(0.5, 1, 16),
      gamma=(0.5, 1, 2, 3, 5),
      normalization=(filter_bank.Normalization.L1, filter_bank.Normalization.L2)
  )
  def test_morse_normalization(self, beta, gamma, normalization):

    def filter_norm(x):
      y = np.abs(
          filter_bank.morse_wavelet(
              beta=beta,
              gamma=gamma,
              domain=filter_bank.Domain.FREQUENCY,
              normalization=normalization)(x))
      if normalization is filter_bank.Normalization.L2:
        y = y**2
      return y

    norm, abserror = sp.integrate.quad(filter_norm, -np.inf, np.inf)
    np.testing.assert_allclose(norm, 1, rtol=1e-3)
    np.testing.assert_allclose(abserror, 0, atol=1e-3)

  @parameterized.product(
      filter_=(filter_bank.gabor_filter, filter_bank.morlet_wavelet,
               filter_bank.sinc_filter),
      sigma=(0.1, 1.0, 3.0),
      domain=(filter_bank.Domain.TIME, filter_bank.Domain.FREQUENCY))
  def test_l2_normalization(self, filter_, sigma, domain):
    if (filter_ is filter_bank.sinc_filter and sigma > 1 and
        domain is filter_bank.Domain.FREQUENCY):
      self.skipTest("unsupported by scipy.integrate.quad")

    def filter_norm(x):
      y = np.abs(
          filter_(
              sigma=sigma,
              domain=domain,
              normalization=filter_bank.Normalization.L2)(x))**2
      return y

    norm, abserror = sp.integrate.quad(filter_norm, -np.inf, np.inf)
    np.testing.assert_allclose(norm, 1, rtol=1e-3)
    np.testing.assert_allclose(abserror, 0, atol=1e-3)

  @parameterized.product(
      filter_=(filter_bank.gabor_filter,), sigma=(0.1, 1.0, 10.0))
  def test_l1_time_normalization(self, filter_, sigma):

    def filter_norm(x):
      y = np.abs(
          filter_(
              sigma=sigma,
              domain=filter_bank.Domain.TIME,
              normalization=filter_bank.Normalization.L1)(x))
      return y

    norm, abserror = sp.integrate.quad(filter_norm, -np.inf, np.inf)
    np.testing.assert_allclose(norm, 1, rtol=1e-3)
    np.testing.assert_allclose(abserror, 0, atol=1e-3)

  @parameterized.product(
      filter_=(filter_bank.gabor_filter,), sigma=(0.1, 1.0, 10.0))
  def test_l1_freq_normalization(self, filter_, sigma):

    freq_filter = filter_(
        sigma=sigma,
        domain=filter_bank.Domain.FREQUENCY,
        normalization=filter_bank.Normalization.L1)

    peak_freq, *_ = sp.optimize.fmin(lambda f: -np.abs(freq_filter(f)), sigma)
    self.assertLessEqual(np.abs(freq_filter(peak_freq)), 1 + 1e-3)

  @parameterized.parameters(filter_bank.Domain.TIME,
                            filter_bank.Domain.FREQUENCY)
  def test_sinc_l1_normalization(self, domain):
    with self.assertRaises(ValueError):
      filter_bank.sinc_filter(
          sigma=1.0, domain=domain, normalization=filter_bank.Normalization.L1)

  @parameterized.parameters(0.1, 1.0, 10.0)
  def test_gabor_peak_freq(self, sigma):

    def neg_gabor_l2_freq(w):
      return -filter_bank.gabor_filter(
          sigma=sigma,
          domain=filter_bank.Domain.FREQUENCY,
          normalization=filter_bank.Normalization.L2)(
              w)

    peak_freq, *_ = sp.optimize.fmin(neg_gabor_l2_freq, 0)
    np.testing.assert_allclose(peak_freq, sigma / (2 * np.pi), 1e-2)

  @parameterized.parameters(filter_bank.Domain.TIME,
                            filter_bank.Domain.FREQUENCY)
  def test_morlet_equivalence(self, domain):
    # For a large sigma, morlet and Gabor should be about the same
    sigma = 6
    xs = np.linspace(-5, 5)
    gabor_filter = filter_bank.gabor_filter(
        sigma=sigma, domain=domain, normalization=filter_bank.Normalization.L2)
    morlet_wavelet = filter_bank.morlet_wavelet(
        sigma=sigma, domain=domain, normalization=filter_bank.Normalization.L2)
    np.testing.assert_allclose(gabor_filter(xs), morlet_wavelet(xs), atol=1e-5)

  @parameterized.parameters((filter_bank.gabor_filter, 1e-3),
                            (filter_bank.morlet_wavelet, 1e-3),
                            (filter_bank.sinc_filter, 0.05))
  def test_domain(self, filter_, atol):
    # We test whether the frequency and time domain match
    # Note that for a function g(t), the DFT in NumPy is like having
    # a_m = g(m / n) where n is the window size (see `numpy.fft` for details),
    # so if we evaluate our kernel on [-L, L] then we need to rescale and shift.

    # Parameters
    sigma = 1.5
    window_length = 4001
    lim = 6

    # The time filter is modulated by a unit normal, so 4 stdevs covers it
    ts = np.linspace(-lim, lim, window_length)
    time_response = filter_(
        sigma=sigma,
        domain=filter_bank.Domain.TIME,
        normalization=filter_bank.Normalization.L2)(
            ts)

    # This ensures that the filter is inside the range of frequencies
    d = 1.0 / lim
    fs = np.fft.fftshift(np.fft.fftfreq(window_length, d))
    freq_response = filter_(
        sigma=sigma,
        domain=filter_bank.Domain.FREQUENCY,
        normalization=filter_bank.Normalization.L2)(
            fs)
    # Shift and scale (scaling the argument is done in interp1d)
    freq_response = freq_response * np.exp(-np.pi * 2j * fs * lim) / (2 * lim)
    # Interpolate to evaluate at the same points as the DFT
    real_freq_response = sp.interpolate.interp1d(fs * 2 * lim,
                                                 np.real(freq_response))

    # DFT of filter in the time domain
    fs_dft = np.fft.fftshift(np.fft.fftfreq(window_length, 1 / window_length))
    freq_response_dft = np.fft.fftshift(
        np.fft.fft(time_response)) / window_length
    overlap = (-1 / d * lim < fs_dft) & (fs_dft < 1 / d * lim)
    fs_dft = fs_dft[overlap]
    np.testing.assert_allclose(
        real_freq_response(fs_dft),
        np.real(freq_response_dft[overlap]),
        atol=atol)

  @parameterized.parameters(filter_bank.Normalization.L1,
                            filter_bank.Normalization.L2)
  def test_domain_filtering(self, normalization):
    # Check that applying a filter in the time domain and frequency domain
    # is the same thing
    num_filters = 3
    signal_length_frames = 4_096
    window_size_frames = 801

    # Construct a non-stationary signal
    ts = np.linspace(0, 2 * np.pi * 73, signal_length_frames)
    signal = np.sin(ts * (1 + np.sin(ts) * 0.1))
    signal = signal[np.newaxis, :, np.newaxis]

    # Sample random filters
    sigma = np.linspace(0.1, 10, num_filters)
    scale_factors = np.linspace(50, 100, num_filters)

    # Apply the filters in the time domain
    time_filter = filter_bank.gabor_filter(sigma, filter_bank.Domain.TIME,
                                           normalization)
    time_filtered_signal = filter_bank.convolve_filter(time_filter, signal,
                                                       scale_factors,
                                                       normalization,
                                                       window_size_frames)

    # Apply the filters in the frequency domain
    freq_filter = filter_bank.gabor_filter(sigma, filter_bank.Domain.FREQUENCY,
                                           normalization)
    freq_filtered_signal = filter_bank.multiply_filter(freq_filter, signal,
                                                       scale_factors,
                                                       normalization)

    # Ignore boundary effects
    middle = slice(window_size_frames // 2, -window_size_frames // 2)

    np.testing.assert_allclose(
        time_filtered_signal[:, middle],
        freq_filtered_signal[:, middle],
        atol=1e-3)

  def test_melspec_params(self):
    # Check that we get the same melspec params as LEAF
    # We use few filters and lots of detail in the frequency domain, otherwise
    # LEAF's implementation has too much noise to compare against
    n_filters = 16
    min_freq = 400.
    max_freq = 20_000.
    sample_rate = 44_800
    window_len = 201
    n_fft = 8_192
    normalize_energy = False

    filters = melfilters.Gabor(
        n_filters=n_filters,
        min_freq=min_freq,
        max_freq=max_freq,
        sample_rate=sample_rate,
        window_len=window_len,
        n_fft=n_fft,
        normalize_energy=normalize_energy)
    center_leaf, fwhm_leaf = tf.transpose(filters.gabor_params_from_mels)

    center, fwhm = filter_bank.melspec_params(n_filters, sample_rate, min_freq,
                                              max_freq)

    np.testing.assert_allclose(center, center_leaf, rtol=1e-2)
    np.testing.assert_allclose(fwhm, fwhm_leaf, rtol=1e-2)

  def test_leaf_compat(self):
    # Check for compatibility with the LEAF codebase

    # Construct a set of Gabor filters
    kernel_size = window_size_frames = 201
    sample_rate = 16_000
    kernel_initializer = initializers.GaborInit(
        sample_rate=sample_rate, min_freq=60.0, max_freq=7_800.0)

    gabor_conv1d_tf = convolution.GaborConv1D(
        filters=16,
        kernel_size=kernel_size,
        strides=1,
        padding="SAME",
        use_bias=False,
        input_shape=(None, None, 1),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        name="gabor_conv1d",
        trainable=True)

    kernel, = gabor_conv1d_tf.get_weights()
    center, fwhm = kernel.T

    # Apply filters to random signal
    ts = tf.linspace(0., 1_000 * 2 * np.pi, sample_rate)
    signal = tf.math.sin(ts * (1 + tf.math.sin(ts) * 0.1))
    signal = signal[tf.newaxis, :, tf.newaxis]
    output_tf = gabor_conv1d_tf.call(signal)
    output_tf = tf.cast(output_tf, tf.complex64)
    output_tf = output_tf[..., ::2] - output_tf[..., 1::2] * 1j

    # Equivalent operation
    sigma = center * fwhm
    gabor_filter = filter_bank.gabor_filter(sigma, filter_bank.Domain.TIME,
                                            filter_bank.Normalization.L1)
    output_jax = filter_bank.convolve_filter(gabor_filter, np.array(signal),
                                             fwhm, filter_bank.Normalization.L1,
                                             window_size_frames)

    np.testing.assert_allclose(output_jax, output_tf, rtol=1e-2)


if __name__ == "__main__":
  absltest.main()
