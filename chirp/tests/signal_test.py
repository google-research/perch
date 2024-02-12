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

"""Tests for chirp.signal."""
from chirp import signal
from jax import numpy as jnp
import numpy as np
import tensorflow as tf

from absl.testing import absltest


class SignalTest(absltest.TestCase):

  def test_linear_to_mel_weight_matrix(self):
    jax_val = signal.linear_to_mel_weight_matrix()
    tf_val = tf.signal.linear_to_mel_weight_matrix()

    np.testing.assert_allclose(jax_val, tf_val, rtol=1e-3)

  def test_frame(self):
    shape = (2, 7, 3)
    signal_ = jnp.reshape(jnp.arange(2 * 7 * 3), shape)
    frames = signal.frame(signal_, 5, 2, axis=1)
    self.assertEqual(frames.shape, (2, 2, 5, 3))

    np.testing.assert_array_equal(frames[1, 1, :, 2], signal_[1, 2:7, 2])


if __name__ == "__main__":
  absltest.main()
