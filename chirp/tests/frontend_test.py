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

"""Tests for frontend."""
from chirp.models import frontend
from jax import numpy as jnp
from jax import random

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
        random.PRNGKey(0), cls.batch_dims + (cls.num_samples,))
    cls.audio = cls.signal + cls.noise

  @parameterized.product(({
      "module_type": frontend.STFT,
      "module_kwargs": {}
  }, {
      "module_type": frontend.MelSpectrogram,
      "module_kwargs": {
          "kernel_size": 128,
          "sample_rate": 11_025,
          "freq_range": (60, 10_000)
      }
  }, {
      "module_type": frontend.LearnedFrontend,
      "module_kwargs": {
          "kernel_size": 256
      }
  }, {
      "module_type": frontend.MorletWaveletTransform,
      "module_kwargs": {
          "kernel_size": 256,
          "sample_rate": 11_025,
          "freq_range": (60, 10_000)
      }
  }),
                         stride=(10, 11))
  def test_output_size(self, module_type, module_kwargs, stride):
    features = 7

    module = module_type(stride=stride, features=features, **module_kwargs)
    variables = module.init(random.PRNGKey(0), self.audio)
    output = module.apply(variables, self.audio)
    self.assertEqual(
        output.shape,
        self.batch_dims + (-(-self.num_samples // stride), features))

  @parameterized.parameters(
      (frontend.STFT, frontend.ISTFT, {}),
      (frontend.LearnedFrontend, frontend.InverseLearnedFrontend, {
          "kernel_size": 256
      }))
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


if __name__ == "__main__":
  absltest.main()
