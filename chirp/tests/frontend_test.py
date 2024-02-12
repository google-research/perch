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

"""Tests for frontend."""
from chirp.models import frontend
from jax import numpy as jnp
from jax import random
from jax.experimental import jax2tf
import numpy as np
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized


class FrontendTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.length_s = 2
    cls.sample_rate_hz = 11_025
    cls.num_samples = cls.sample_rate_hz * cls.length_s
    cls.batch_dims = (2, 3)
    cls.signal = jnp.sin(jnp.linspace(0.0, 440 * jnp.pi, cls.num_samples))
    cls.noise = 0.5 * random.normal(
        random.PRNGKey(0), cls.batch_dims + (cls.num_samples,)
    )
    cls.audio = cls.signal + cls.noise

  @parameterized.product(
      (
          {
              "module_type": frontend.STFT,
              "module_kwargs": {},
          },
          {
              "module_type": frontend.MelSpectrogram,
              "module_kwargs": {
                  "kernel_size": 128,
                  "sample_rate": 11_025,
                  "freq_range": (60, 5_000),
              },
          },
          {
              "module_type": frontend.SimpleMelspec,
              "module_kwargs": {
                  "kernel_size": 128,
                  "sample_rate": 11_025,
                  "freq_range": (60, 5_000),
              },
          },
          {
              "module_type": frontend.LearnedFrontend,
              "module_kwargs": {
                  "kernel_size": 256,
              },
          },
          {
              "module_type": frontend.MorletWaveletTransform,
              "module_kwargs": {
                  "kernel_size": 256,
                  "sample_rate": 11_025,
                  "freq_range": (60, 10_000),
              },
          },
      ),
      stride=(10, 11),
  )
  def test_output_size(self, module_type, module_kwargs, stride):
    features = 7

    module = module_type(stride=stride, features=features, **module_kwargs)
    variables = module.init(random.PRNGKey(0), self.audio)
    output = module.apply(variables, self.audio)
    self.assertEqual(
        output.shape,
        self.batch_dims + (-(-self.num_samples // stride), features),
    )

  @parameterized.parameters(
      (frontend.STFT, frontend.ISTFT, {}),
      (
          frontend.LearnedFrontend,
          frontend.InverseLearnedFrontend,
          {
              "kernel_size": 256,
          },
      ),
  )
  def test_inverse(self, module_type, inverse_module_type, module_kwargs):
    stride = 10
    features = 7

    module = module_type(stride=stride, features=features, **module_kwargs)
    variables = module.init(random.PRNGKey(0), self.audio)
    output = module.apply(variables, self.audio)

    inverse_module = inverse_module_type(stride=stride, **module_kwargs)
    inverse_variables = inverse_module.init(random.PRNGKey(0), output)
    inversed = inverse_module.apply(inverse_variables, output)

    self.assertEqual(self.audio.shape, inversed.shape)

  @parameterized.parameters(
      {
          "module_type": frontend.STFT,
          "module_kwargs": {
              "features": 129,  # Note: Required that f-1=2**k for some k.
              "stride": 64,
          },
      },
      {
          "module_type": frontend.MelSpectrogram,
          "module_kwargs": {
              "features": 32,
              "stride": 64,
              "kernel_size": 64,
              "sample_rate": 22_025,
              "freq_range": (60, 10_000),
          },
          "atol": 1e-4,
      },
      {
          "module_type": frontend.SimpleMelspec,
          "module_kwargs": {
              "features": 32,
              "stride": 64,
              "kernel_size": 64,
              "sample_rate": 22_025,
              "freq_range": (60, 10_000),
          },
          "atol": 1e-4,
      },
      {
          "module_type": frontend.LearnedFrontend,
          "module_kwargs": {
              "features": 32,
              "stride": 64,
              "kernel_size": 64,
          },
      },
      {
          "module_type": frontend.ISTFT,
          "module_kwargs": {
              "stride": 64,
          },
          "signal_shape": (1, 25, 64),
      },
      {
          "module_type": frontend.InverseLearnedFrontend,
          "module_kwargs": {
              "stride": 32,
              "kernel_size": 64,
          },
          "signal_shape": (1, 25, 64),
      },
  )
  def test_tflite_stft_export(
      self, module_type, module_kwargs, signal_shape=None, atol=1e-6
  ):
    # Note that the TFLite stft requires power-of-two nfft, given by:
    # nfft = 2 * (features - 1).
    if signal_shape is None:
      signal = self.audio
    else:
      signal = jnp.zeros(signal_shape, jnp.float32)
    fe = module_type(**module_kwargs)
    params = fe.init(random.PRNGKey(0), signal)

    tf_predict = tf.function(
        jax2tf.convert(
            lambda signal: fe.apply(params, signal), enable_xla=False
        ),
        input_signature=[
            tf.TensorSpec(shape=signal.shape, dtype=tf.float32, name="input")
        ],
        autograph=False,
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_predict.get_concrete_function()], tf_predict
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    tflite_float_model = converter.convert()

    # Use the converted TFLite model.
    interpreter = tf.lite.Interpreter(model_content=tflite_float_model)
    interpreter.allocate_tensors()
    input_tensor = interpreter.get_input_details()[0]
    output_tensor = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_tensor["index"], signal)
    interpreter.invoke()
    output_tflite = interpreter.get_tensor(output_tensor["index"])

    # Check approximate agreement of TFLite output with the jax function.
    output_jax = fe.apply(params, signal)
    np.testing.assert_allclose(output_tflite, output_jax, atol=1e-4)

  def test_simple_melspec(self):
    frontend_args = {
        "features": 32,
        "stride": 64,
        "kernel_size": 64,
        "sample_rate": 22_025,
        "freq_range": (60, 10_000),
    }
    simple_mel = frontend.SimpleMelspec(**frontend_args)
    simple_mel_params = simple_mel.init(random.PRNGKey(0), self.audio)
    got_simple = simple_mel.apply(simple_mel_params, self.audio)

    # Check that export works without SELECT_TF_OPS.
    tf_predict = tf.function(
        jax2tf.convert(
            lambda signal: simple_mel.apply(simple_mel_params, signal),
            enable_xla=False,
        ),
        input_signature=[
            tf.TensorSpec(
                shape=self.audio.shape, dtype=tf.float32, name="input"
            )
        ],
        autograph=False,
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_predict.get_concrete_function()], tf_predict
    )
    converter.target_spec.supported_ops = [
        # Look, ma, no tf.lite.OpsSet.SELECT_TF_OPS!
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    ]
    tflite_float_model = converter.convert()

    # Use the converted TFLite model.
    interpreter = tf.lite.Interpreter(model_content=tflite_float_model)
    interpreter.allocate_tensors()
    input_tensor = interpreter.get_input_details()[0]
    output_tensor = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_tensor["index"], self.audio)
    interpreter.invoke()
    output_tflite = interpreter.get_tensor(output_tensor["index"])

    # Check approximate agreement of TFLite output with the jax function.
    np.testing.assert_allclose(output_tflite, got_simple, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
