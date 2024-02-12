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

"""Masked autoencoder for spectrograms."""


import flax.linen as nn
from jax import numpy as jnp
from jax import random
import optax
from scenic.model_lib.layers import attention_layers
from scenic.projects.baselines import vit


def get_patches(x: jnp.ndarray, patch_size: tuple[int, int]) -> jnp.ndarray:
  """Split input into patches.

  Args:
    x: The input to split into patches. Must be of size (..., height, width,
      channels).
    patch_size: The size of the patches, (patch height, patch width).

  Returns:
    The patches of x as size (..., height / patch height, width / patch width,
    patch height, patch width).
  """
  # For a single batch dimension, should be equivalent to:
  # lax.conv_general_dilated_patches(x, patch_size, patch_size, 'VALID',
  #                                  dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
  *b, h, w, c = x.shape
  ph, pw = patch_size
  if h % ph != 0 or w % pw != 0:
    raise ValueError('patch size does not divide image size')
  x = jnp.reshape(x, b + [h // ph, ph, w // pw, pw, c])
  return jnp.swapaxes(x, -3, -4)


def merge_patches(x: jnp.ndarray) -> jnp.ndarray:
  """Reshape patched image into single image.

  Args:
    x: The patches of size (..., height, width, patch height, patch width,
      channels).

  Returns:
    The image of size (..., height * patch height, width * patch width,
    channels).
  """
  *b, h, w, ph, pw, c = x.shape
  x = jnp.swapaxes(x, -3, -4)
  return jnp.reshape(x, b + [h * ph, w * pw, c])


class Encoder(nn.Module):
  """Encode patches.

  Following the Masked Spectrogram Modelling paper this uses a ViT-B encoder.
  We add 1D sinusoidal embeddings to the patches before masking out a fraction
  of patches at random, only encoding the unmasked ones.

  This is the same encoder as used by the MAEs that Listen paper. The MAE-AST
  paper uses a slight variation that only has half the number of layers (6).

  Attributes:
    mlp_dim: Dimension of the MLP on top of attention block.
    num_layers: Number of transformer layers.
    num_heads: Number of self-attention heads.
    patch_size: Size of the patches (as a tuple). Patches are non-overlapping.
    mask_rate: The fraction of patches to mask.
    hidden_size: Size of the linear embedding of the patches.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: The layer dropout probability.
    class_token: Whether or not to prepend a zero-initialized learned class
      token to the patches.
  """

  patch_size: tuple[int, int] = (16, 16)
  mlp_dim: int = 3072
  num_layers: int = 12
  num_heads: int = 12
  mask_rate: float = 0.75
  hidden_size: int = 768
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.0
  class_token: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool):
    """Mask the patches and encode the remaining ones.

    Note that the indices of masked and unmasked patches can be used as
    follows, where x is an array of size (batch, patches, ...):

      jnp.take_along_axis(x, indices[..., jnp.newaxis], axis=1)

    or

      x.at[jnp.arange(n)[:, jnp.newaxis], indices]

    Args:
      x: An image/spectrogram.
      train: Whether or not this is training.

    Returns:
      The encoded patches and the indices of the unmasked and masked patches.
      The encoded patches are of the shape (batch, unmasked patches, features).
    """
    if jnp.ndim(x) != 4:
      raise ValueError('x must be of shape (batch, height, width, channels')

    fh, fw = self.patch_size
    # Extracting patches and then embedding is in fact a single convolution.
    x = nn.Conv(
        self.hidden_size,
        (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding',
    )(x)
    n, h, w, c = x.shape

    # Add positional embeddings
    x = jnp.reshape(x, [n, h * w, c])
    x = attention_layers.Add1DPositionEmbedding(posemb_init=None)(x)

    num_patches = h * w
    indices = jnp.tile(jnp.arange(h * w), (n, 1))
    if train:
      num_patches = int(num_patches * (1 - self.mask_rate))
      rng = self.make_rng('patch_mask')
      indices = random.permutation(rng, indices, axis=1, independent=True)
    unmasked, masked = indices[:, :num_patches], indices[:, num_patches:]
    x = jnp.take_along_axis(x, unmasked[..., jnp.newaxis], axis=1)

    # If we want to add a class token, add it here.
    if self.class_token:
      class_token = self.param(
          'class_token', nn.initializers.zeros, (1, 1, c), x.dtype
      )
      class_token = jnp.tile(class_token, [n, 1, 1])
      x = jnp.concatenate([class_token, x], axis=1)

    x = vit.Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        positional_embedding='none',
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
    )(x, train=train)

    return x, unmasked, masked


class Embedder(nn.Module):
  encoder: Encoder = Encoder()
  taxonomy_loss_weight: float = 0.0

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      *,
      train: bool,
      use_running_average: bool | None = None
  ):
    encoded_patches, _, _ = self.encoder(x, train=train)
    embedding = jnp.mean(encoded_patches, axis=-2)
    return optax.scale_gradient(embedding, 0.01)


class Decoder(nn.Module):
  """Decode patches.

  This decoder follows the Masked Spectrogram Modeling paper, which follows
  ViT-S (384 dimensions, 6 heads) with a reduced number of layers (4).

  The MAE-AST paper uses ViT-B instead (12 heads, 768 dimensions) but with only
  2 layers.

  The MAEs that Listen paper uses a deeper, custom decoder (8 layers when using
  global attention) with 512 dimensions.

  Attributes:
    output_size: The size of the output image (height, width, channels).
    patch_size: The size of each patch (height and width).
    mlp_dim: Dimension of the MLP on top of attention block.
    num_layers: Number of transformer layers.
    num_heads: Number of self-attention heads.
    hidden_size: Size of the linear embedding of the patches.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: The layer dropout probability.
  """

  output_size: tuple[int, int, int]
  patch_size: tuple[int, int] = (16, 16)
  mlp_dim: int = 1536
  num_layers: int = 4
  num_heads: int = 6
  hidden_size: int = 384
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(self, x: jnp.ndarray, unmasked: jnp.ndarray, *, train: bool):
    """Decode the patches.

    Args:
      x: The embeddings of the unmasked patches.
      unmasked: The indices of the unmasked patches in the spectrogram.
      train: Whether this is training time.

    Returns:
      The decoded patches, of the form (batch, patches, features). The number
      of features is equal to the patch size times number of channels.
    """
    # First restore the patches in their correct order and use mask tokens
    n, num_patches, features = x.shape
    h, w, c = (
        self.output_size[0] // self.patch_size[0],
        self.output_size[1] // self.patch_size[1],
        self.output_size[2],
    )
    if unmasked.shape != (n, num_patches):
      raise ValueError('shape of encoded patches and mask do not match')
    mask_token = self.param(
        'mask_token', nn.initializers.zeros, (1, 1, features), x.dtype
    )
    embeddings = jnp.tile(mask_token, (n, h * w, 1))
    embeddings = embeddings.at[jnp.arange(n)[:, jnp.newaxis], unmasked].set(x)

    if features != self.hidden_size:
      x = nn.Dense(features=self.hidden_size)(embeddings)

    # Transformer decoder
    x = vit.Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        positional_embedding='sinusoidal_1d',
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
    )(x, train=train)

    x = nn.Dense(features=self.patch_size[0] * self.patch_size[1] * c)(x)

    return x


class MaskedAutoencoder(nn.Module):
  """A masked autoencoder."""

  encoder: nn.Module
  decoder: nn.Module

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool):
    """Apply masked autoencoder.

    Args:
      x: An image of size (batch, height, width, channels).
      train: Whether this is training.

    Returns:
      The decoded patches (of shape (batch, patches, features)).
    """
    if self.encoder.patch_size != self.decoder.patch_size:
      raise ValueError('patch sizes do not match')
    encoded_patches, unmasked, masked = self.encoder(x, train=train)
    decoded_patches = self.decoder(encoded_patches, unmasked, train=train)
    patches = get_patches(x, self.encoder.patch_size)
    patches = jnp.reshape(patches, decoded_patches.shape)
    return decoded_patches, patches, masked
