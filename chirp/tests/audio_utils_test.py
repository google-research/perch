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
from jax.experimental import jax2tf
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
        random.PRNGKey(0), cls.batch_dims + (cls.num_frames,))
    cls.audio = cls.signal + cls.noise

    _, _, cls.spectrogram = jsp.signal.stft(cls.audio)

  # NOTE: We don't test with odd values for nperseg or with a Hamming window
  # since TF numerics are quite noisy in those cases
  @parameterized.product(
      fs=(1.0, 3.4),
      noverlap=(None, 23),
      nfft=(None, 263),  # nfft must be >= nperseg
      boundary=(None, "zeros", "constant"),
      padded=(True, False),
      axis=(-1, 1))
  def test_stft(self, **kwargs):
    audio = jnp.swapaxes(self.audio, -1, kwargs["axis"])

    f_jax, t_jax, stft_jax = jsp.signal.stft(audio, **kwargs)
    f_tf, t_tf, stft_tf = audio_utils.stft(audio, **kwargs)

    np.testing.assert_allclose(f_tf, f_jax, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(t_tf, t_jax, rtol=1e-5, atol=1e-6)
    rtol, atol = 1e-2, 1e-3
    np.testing.assert_allclose(stft_tf, stft_jax, rtol=rtol, atol=atol)

  @parameterized.product(
      noverlap=(None, 23), nfft=(None, 263), boundary=(True,))
  def test_istft(self, **kwargs):

    _, istft_jax = jsp.signal.istft(self.spectrogram, **kwargs)
    _, istft_tf = audio_utils.istft(self.spectrogram, **kwargs)

    np.testing.assert_allclose(istft_tf, istft_jax, rtol=1e-2, atol=1e-3)

  def test_tflite_melspec(self):
    # Demonstrate TFLite export of the melspec computation.

    # Export a TFLite model from the melspec model function.
    tf_predict = tf.function(
        jax2tf.convert(
            lambda audio: audio_utils.stft(audio)[2], enable_xla=False),
        input_signature=[
            tf.TensorSpec(
                shape=self.audio.shape, dtype=tf.float32, name="input")
        ],
        autograph=False)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_predict.get_concrete_function()], tf_predict)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_float_model = converter.convert()

    # Use the converted TFLite model.
    interpreter = tf.lite.Interpreter(model_content=tflite_float_model)
    interpreter.allocate_tensors()
    input_tensor = interpreter.get_input_details()[0]
    output_tensor = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_tensor["index"], self.audio)
    interpreter.invoke()
    output_array = interpreter.get_tensor(output_tensor["index"])

    # Check approximate agreement of TFLite output with the jax function.
    melspec_jax = audio_utils.stft(self.audio)[2]
    np.testing.assert_allclose(output_array, melspec_jax, atol=1e-6)

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
        eps=eps)[0]
    librosa_out = spectrum.pcen(
        spec,
        b=smoothing_coef,
        gain=gain,
        bias=bias,
        power=1 / root,
        eps=eps,
        # librosa starts with an initial state of (1 - s), we start with x[0]
        zi=(1 - smoothing_coef) * spec[..., 0:1, :],
        axis=-2)

    np.testing.assert_allclose(out, librosa_out, rtol=5e-2)

  def test_pad_to_length_if_shorter(self):
    audio = jnp.asarray([-1, 0, 1, 0], dtype=jnp.float32)
    np.testing.assert_allclose(
        audio_utils.pad_to_length_if_shorter(audio, 4), audio)
    np.testing.assert_allclose(
        audio_utils.pad_to_length_if_shorter(audio, 6),
        jnp.asarray([0, -1, 0, 1, 0, -1], dtype=jnp.float32))


if __name__ == "__main__":
  absltest.main()
