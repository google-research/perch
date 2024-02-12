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

"""Tests for MAE."""

from chirp.models import mae
from jax import numpy as jnp
from jax import random
from absl.testing import absltest


class MaeTest(absltest.TestCase):

  def test_shapes(self):
    batch_size = 16
    image_size = (256, 512)
    patch_size = (16, 16)
    mask_rate = 0.8
    hidden_size = 64
    c = 2

    h, w = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
    num_patches = int(h * w * (1 - mask_rate))

    inputs = jnp.ones((batch_size,) + image_size + (c,))
    rng = random.PRNGKey(0)
    params_rng, mask_rng, dropout_rng = random.split(rng, num=3)
    encoder = mae.Encoder(
        mlp_dim=32,
        num_layers=2,
        num_heads=8,
        patch_size=patch_size,
        mask_rate=mask_rate,
        hidden_size=hidden_size,
    )
    (encoded_patches, unmasked, masked), _ = encoder.init_with_output(
        {"params": params_rng, "patch_mask": mask_rng, "dropout": dropout_rng},
        inputs,
        train=True,
    )

    self.assertEqual(
        encoded_patches.shape, (batch_size, num_patches, hidden_size)
    )
    self.assertEqual(unmasked.shape, (batch_size, num_patches))
    self.assertEqual(masked.shape, (batch_size, h * w - num_patches))

    decoder = mae.Decoder(
        output_size=image_size + (c,),
        patch_size=patch_size,
        mlp_dim=32,
        num_layers=2,
        num_heads=8,
    )
    decoded_patches, _ = decoder.init_with_output(
        {"params": params_rng, "dropout": dropout_rng},
        encoded_patches,
        unmasked,
        train=True,
    )
    self.assertEqual(
        decoded_patches.shape,
        (batch_size, h * w, patch_size[0] * patch_size[1] * c),
    )


if __name__ == "__main__":
  absltest.main()
