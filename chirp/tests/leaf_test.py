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

"""Tests for LEAF."""
from chirp.models import leaf
from jax import numpy as jnp
from leaf_audio import convolution
from leaf_audio import impulse_responses
from leaf_audio import initializers
import numpy as np

from absl.testing import absltest


class LeafTest(absltest.TestCase):

  def test_gabor_init(self):
    n_filters = 32
    n_fft = 128
    sample_rate = 22_050
    min_freq = 60.
    max_freq = 10_000.

    tf_initializer = initializers.GaborInit(
        n_filters=n_filters,
        n_fft=n_fft,
        sample_rate=sample_rate,
        min_freq=min_freq,
        max_freq=max_freq)

    kernel_tf = tf_initializer((n_filters, 2))
    kernel = leaf.gabor_init(n_filters, n_fft, sample_rate, min_freq, max_freq)

    np.testing.assert_allclose(kernel_tf, kernel)

  def test_gabor_impulse_response(self):
    t = np.arange(-10., 10.)
    center = np.random.randn(8)
    fwhm = np.random.rand(8)

    filters_tf = impulse_responses.gabor_impulse_response(t, center, fwhm)
    filters = leaf.gabor_impulse_response(t, center, fwhm)

    np.testing.assert_allclose(filters_tf, filters, rtol=1e-5)

  def test_gabor_conv1d(self):
    kernel_size = 32
    kernel_initializer = initializers.GaborInit(
        sample_rate=16_000, min_freq=60.0, max_freq=7_800.0)
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

    signal = np.random.randn(1, 100, 1).astype(np.float32)
    output_tf = gabor_conv1d_tf.call(signal)
    output = leaf.gabor_conv1d(jnp.array(signal), kernel, kernel_size)
    np.testing.assert_allclose(output, output_tf, 1e-2)


if __name__ == "__main__":
  absltest.main()
