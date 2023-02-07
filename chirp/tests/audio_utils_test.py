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

"""Tests for audio utilities."""

from chirp import audio_utils
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp
from librosa.core import spectrum
import numpy as np
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized


class AudioUtilsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.length_s = 2
    cls.sample_rate_hz = 11_025
    cls.num_frames = cls.sample_rate_hz * cls.length_s
    cls.batch_dims = (2, 3)
    cls.signal = jnp.sin(jnp.linspace(0.0, 440 * jnp.pi, cls.num_frames))
    cls.noise = 0.5 * random.normal(
        random.PRNGKey(0), cls.batch_dims + (cls.num_frames,)
    )
    cls.audio = cls.signal + cls.noise

    _, _, cls.spectrogram = jsp.signal.stft(cls.audio)

  def test_pcen(self):
    gain = 0.5
    smoothing_coef = 0.1
    bias = 2.0
    root = 2.0
    eps = 1e-6

    spec = jnp.abs(self.spectrogram)

    out = audio_utils.pcen(
        spec,
        gain=gain,
        smoothing_coef=smoothing_coef,
        bias=bias,
        root=root,
        eps=eps,
    )[0]
    librosa_out = spectrum.pcen(
        spec,
        b=smoothing_coef,
        gain=gain,
        bias=bias,
        power=1 / root,
        eps=eps,
        # librosa starts with an initial state of (1 - s), we start with x[0]
        zi=(1 - smoothing_coef) * spec[..., 0:1, :],
        axis=-2,
    )

    np.testing.assert_allclose(out, librosa_out, rtol=5e-2)

  @parameterized.product(
      # NOTE: TF and JAX have different outputs when nperseg is odd.
      nperseg=(256, 230),
      noverlap=(0, 17),
      # NOTE: FFT length must be factorizable into primes less than 127 (this
      # is a cuFFT restriction).
      nfft=(256, 301),
      boundary=("zeros", None),
      padded=(True, False),
  )
  def test_stft_tf(self, nperseg, noverlap, nfft, boundary, padded):
    batch_size = 3
    sample_rate_hz = 22050
    window = "hann"
    # NOTE: We don't test the Hamming window, since TensorFlow and SciPy have
    # different implementations, which leads to slightly different results.
    # To be precise, the difference is that:
    # sp.signal.get_window("hamming", N) == tf.signal.hamming_window(N + 1)[:-1]

    time_size = 5 * sample_rate_hz
    audio = jnp.sin(jnp.linspace(0.0, 440 * jnp.pi, time_size))
    noise = 0.01 * random.normal(random.PRNGKey(0), (batch_size, time_size))
    signal = audio + noise

    _, _, stfts = jsp.signal.stft(
        signal,
        fs=1 / sample_rate_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=boundary,
        padded=padded,
    )
    stfts_tf = audio_utils.stft_tf(
        tf.constant(signal),
        fs=1 / sample_rate_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=boundary,
        padded=padded,
    )

    np.testing.assert_allclose(stfts, stfts_tf.numpy(), atol=1e-5)

  def test_pad_to_length_if_shorter(self):
    audio = jnp.asarray([-1, 0, 1, 0], dtype=jnp.float32)
    np.testing.assert_allclose(
        audio_utils.pad_to_length_if_shorter(audio, 4), audio
    )
    np.testing.assert_allclose(
        audio_utils.pad_to_length_if_shorter(audio, 6),
        jnp.asarray([0, -1, 0, 1, 0, -1], dtype=jnp.float32),
    )


if __name__ == "__main__":
  absltest.main()
