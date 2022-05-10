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
from jax.experimental import jax2tf
from librosa.core import spectrum
import numpy as np
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized


class AudioUtilsTest(parameterized.TestCase):

  def test_compute_melspec(self):
    sample_rate_hz = 22050
    audio = jnp.sin(jnp.linspace(0.0, 440 * jnp.pi, sample_rate_hz))
    noise = 0.01 * random.normal(random.PRNGKey(0), (2, 3, sample_rate_hz))

    kwargs = {
        "sample_rate_hz": 22050,
        "melspec_depth": 160,
        "melspec_frequency": 100,
        "frame_length_secs": 0.08,
        "upper_edge_hz": 8000.0,
        "lower_edge_hz": 60.0
    }

    # Test mel-spectrogram shape and batching
    melspec = audio_utils.compute_melspec(
        audio + noise[0, 0], scaling_config=None, **kwargs)

    self.assertEqual(melspec.shape, (100, 160))

    batch_melspec = audio_utils.compute_melspec(
        audio + noise, scaling_config=None, **kwargs)

    self.assertEqual(batch_melspec.shape, (2, 3, 100, 160))
    np.testing.assert_allclose(batch_melspec[0, 0], melspec, 1e-6)

    # Test normalization
    melspec = audio_utils.compute_melspec(
        audio + noise[0, 0],
        scaling_config=audio_utils.LogScalingConfig(),
        **kwargs)
    self.assertEqual(melspec.shape, (100, 160))

    melspec = audio_utils.compute_melspec(
        audio + noise[0, 0],
        scaling_config=audio_utils.PCENScalingConfig(),
        **kwargs)
    self.assertEqual(melspec.shape, (100, 160))

  @parameterized.named_parameters(
      {
          "testcase_name": "_raw",
          "scaling_config": None,
          "atol": 2e-3
      }, {
          "testcase_name": "_log",
          "scaling_config": audio_utils.LogScalingConfig(),
          "atol": 2e-4
      }, {
          "testcase_name": "_pcen",
          "scaling_config": audio_utils.PCENScalingConfig(),
          "atol": 4e-4
      })
  def test_jax_tf_equivalence(self, scaling_config, atol):
    sample_rate_hz = 22050
    audio = jnp.sin(jnp.linspace(0.0, 440 * jnp.pi, sample_rate_hz))
    noise = 0.01 * random.normal(random.PRNGKey(0), (2, 3, sample_rate_hz))

    kwargs = {
        "sample_rate_hz": 22050,
        "melspec_depth": 160,
        "melspec_frequency": 100,
        "frame_length_secs": 0.08,
        "upper_edge_hz": 8000.0,
        "lower_edge_hz": 60.0
    }
    melspec_jax = audio_utils.compute_melspec(
        audio + noise[0, 0],
        scaling_config=scaling_config,
        use_tf_stft=False,
        **kwargs)
    melspec_tf = audio_utils.compute_melspec(
        audio + noise[0, 0],
        scaling_config=scaling_config,
        use_tf_stft=True,
        **kwargs)
    np.testing.assert_allclose(melspec_jax, melspec_tf, atol=atol)

  def test_tflite_melspec(self):
    # Demonstrate TFLite export of the melspec computation.
    sample_rate_hz = 22050
    batch_size = 3
    time_size = 5 * sample_rate_hz
    audio = jnp.sin(jnp.linspace(0.0, 440 * jnp.pi, time_size))
    noise = 0.01 * random.normal(random.PRNGKey(0), (batch_size, time_size))

    def melspec_model_fn(audio):
      scaling_config = audio_utils.PCENScalingConfig()
      msf = audio_utils.compute_melspec(
          audio,
          sample_rate_hz=sample_rate_hz,
          melspec_depth=160,
          melspec_frequency=100,
          use_tf_stft=True,
          scaling_config=scaling_config)
      return msf

    # Export a TFLite model from the melspec model function.
    tf_predict = tf.function(
        jax2tf.convert(melspec_model_fn, enable_xla=False),
        input_signature=[
            tf.TensorSpec(
                shape=[batch_size, time_size], dtype=tf.float32, name="input")
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
    interpreter.set_tensor(input_tensor["index"], audio + noise)
    interpreter.invoke()
    output_array = interpreter.get_tensor(output_tensor["index"])[0]

    # Check approximate agreement of TFLite output with the jax function.
    melspec_jax = melspec_model_fn(audio + noise)
    np.testing.assert_allclose(output_array, melspec_jax[0], 5e-3)

  def test_fixed_pcen(self):
    sample_rate_hz = 22050
    audio = jnp.sin(jnp.linspace(0.0, 440 * jnp.pi, sample_rate_hz))
    noise = 0.01 * random.normal(random.PRNGKey(0), (
        1,
        sample_rate_hz,
    ))
    filterbank_energy = audio_utils.compute_melspec(
        audio + noise[0, 0],
        sample_rate_hz=sample_rate_hz,
        melspec_depth=160,
        melspec_frequency=100,
        frame_length_secs=0.08,
        upper_edge_hz=8000.0,
        lower_edge_hz=60.0,
        scaling_config=None)

    gain = 0.5
    smoothing_coef = 0.1
    bias = 2.0
    root = 2.0
    eps = 1e-6

    out = audio_utils.fixed_pcen(
        filterbank_energy,
        gain=gain,
        smoothing_coef=smoothing_coef,
        bias=bias,
        root=root,
        eps=eps)[0]
    librosa_out = spectrum.pcen(
        filterbank_energy,
        b=smoothing_coef,
        gain=gain,
        bias=bias,
        power=1 / root,
        eps=eps,
        # librosa starts with an initial state of (1 - s), we start with x[0]
        zi=[(1 - smoothing_coef) * filterbank_energy[..., 0, :]],
        axis=-2)

    np.testing.assert_allclose(out, librosa_out, rtol=1e-2)


if __name__ == "__main__":
  absltest.main()
