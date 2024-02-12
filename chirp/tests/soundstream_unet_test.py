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

"""Tests for EfficientNet."""
import operator
from chirp.models import soundstream_unet
from jax import numpy as jnp
from jax import random
from jax import tree_util

from absl.testing import absltest


class SoundstreamUNetTest(absltest.TestCase):

  def test_soundstream_unet(self):
    batch_size = 2
    input_time_steps = 16
    input_width = 8

    model = soundstream_unet.SoundstreamUNet(
        base_filters=2,
        bottleneck_filters=4,
        output_filters=8,
        strides=(2, 2),
        feature_mults=(2, 2),
        groups=(1, 2),
    )
    inp_audio = jnp.zeros([batch_size, input_time_steps, input_width])

    (out, embedding), variables = model.init_with_output(
        {"params": random.PRNGKey(0)}, inp_audio, train=True
    )
    self.assertEqual(out.shape, inp_audio.shape)
    # Embedding shape: (batch, input_time / prod(strides), bottleneck_filters).
    self.assertEqual(embedding.shape, (2, 4, 4))

    num_parameters = tree_util.tree_reduce(
        operator.add, tree_util.tree_map(jnp.size, variables["params"])
    )
    self.assertEqual(num_parameters, 864)


if __name__ == "__main__":
  absltest.main()
