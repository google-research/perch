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

"""Separation model."""

from typing import Callable, Sequence

from flax import linen as nn
import jax
from jax import numpy as jnp

SOUNDSTREAM_UNET = "soundstream_unet"


def enforce_mixture_consistency_time_domain(mixture_waveforms,
                                            separated_waveforms,
                                            use_mag_weighting=False):
  """Projection implementing mixture consistency in time domain.

  This projection makes the sum across sources of separated_waveforms equal to
  mixture_waveforms and minimizes the unweighted mean-squared error between the
  sum across sources of separated_waveforms and mixture_waveforms. See
  https://arxiv.org/abs/1811.08521 for the derivation.

  Args:
    mixture_waveforms: Array of mixture waveforms with shape [B, T].
    separated_waveforms: Array of separated waveforms with shape [B, C, T].
    use_mag_weighting: If True, mix weights are magnitude-squared of the
      separated signal.

  Returns:
    Projected separated_waveforms as an array in source image format.
  """
  # Modify the source estimates such that they sum up to the mixture, where
  # the mixture is defined as the sum across sources of the true source
  # targets. Uses the least-squares solution under the constraint that the
  # resulting source estimates add up to the mixture.
  num_sources = separated_waveforms.shape[1]
  mix = jnp.expand_dims(mixture_waveforms, 1)
  mix_estimate = jnp.sum(separated_waveforms, 1, keepdims=True)
  if use_mag_weighting:
    mix_weights = 1e-8 + jnp.mean(separated_waveforms**2, axis=2, keepdims=True)
    mix_weights /= jnp.sum(mix_weights, axis=1, keepdims=True)
  else:
    mix_weights = 1.0 / num_sources
  correction = mix_weights * (mix - mix_estimate)
  separated_waveforms = separated_waveforms + correction
  return separated_waveforms


class SeparationModel(nn.Module):
  """Audio separation model.

  We use a general masked separation approach, similar to ConvTasNet. Input
  audio with shape [[B]atch, [T]ime] is run through an invertible 'bank'
  transform (usually STFT or a learned filterbank), obtaining shape
  [B, T, [F]ilters]. A mask-generator network consumes the banked audio and
  produces a set of (usually sigmoid) masks with shape [B, T, [C]hannels, F].
  These masks are broadcast multiplied by the banked audio to create C separated
  audio channels. Then an 'unbank' (ie, synthesis filterbank) tranformation
  returns the masked audio channels to the time domain. Finally, a mixture
  consistency projection is applied.

  [^1]: ConvTasNet: https://arxiv.org/pdf/1809.07454.pdf

  Attributes:
    bank_transform: A transform consuming a batch of audio with shape [B, T] and
      returning an array of shape [B, T, F].
    unbank_transform: A transform returning an array of shape [B, T, F] to
      time-domain audio with shape [B, T].
    mask_generator: A network transforming an array banked audio of shape [B, T,
      F] to an output with the same batch and time dimensions as the input
      banked audio. This module handles the transformation of the mask_generator
      outputs to actual mask values.
    num_mask_channels: Number of separated channels.
    mask_kernel_size: Kernel size for transpose convolution to mask logits.
    bank_is_real: Indicates if the banked audio is complex valued. If so, we
      take the magnitude of the bank values before feeding them to the
      mask_generator network.
  """

  bank_transform: Callable[[jnp.ndarray], jnp.ndarray]
  unbank_transform: Callable[[jnp.ndarray], jnp.ndarray]
  mask_generator: nn.Module
  num_mask_channels: int = 4
  mask_kernel_size: int = 3
  bank_is_real: bool = False

  def check_shapes(self, banked_inputs, mask_hiddens):
    if mask_hiddens.shape[-3] != banked_inputs.shape[-3]:
      raise ValueError(
          "Output mask_hiddens must have the same time dimensionality as the "
          "banked_inputs. Got shapes: %s vs %s" %
          (mask_hiddens.shape, banked_inputs.shape))

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> Sequence[jnp.ndarray]:
    """Apply the separation model."""
    banked_inputs = self.bank_transform(inputs)
    num_banked_filters = banked_inputs.shape[-1]

    if self.bank_is_real:
      mask_inputs = banked_inputs
    else:
      mask_inputs = jnp.abs(banked_inputs)
    mask_hiddens, _ = self.mask_generator(mask_inputs, train=train)
    self.check_shapes(banked_inputs, mask_hiddens)

    # Convert mask_hiddens to actual mask values.
    # TODO(tomdenton): Check whether non-trivial mask_kernel_size really helps.
    masks = nn.ConvTranspose(
        features=self.num_mask_channels * num_banked_filters,
        kernel_size=(self.mask_kernel_size,))(
            mask_hiddens)
    masks = jax.nn.sigmoid(masks)

    # Reshape the masks for broadcasting to [B, T, C, F].
    masks = jnp.reshape(masks, [
        masks.shape[0], masks.shape[1], self.num_mask_channels,
        num_banked_filters
    ])

    # Apply the masks to the banked input.
    masked_banked_inputs = masks * jnp.expand_dims(banked_inputs, -2)
    # To undo the bank transform, swap axes to get shape [B, C, T, F]
    masked_banked_inputs = jnp.swapaxes(masked_banked_inputs, -2, -3)
    unbanked = self.unbank_transform(masked_banked_inputs)
    unbanked = enforce_mixture_consistency_time_domain(inputs, unbanked)
    return unbanked
