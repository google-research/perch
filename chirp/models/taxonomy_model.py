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

"""Taxonomy model."""
import dataclasses

from chirp.models import conformer
from chirp.models import frontend
from chirp.models import layers
from flax import linen as nn
from jax import numpy as jnp


class TaxonomyModel(nn.Module):
  """Taxonomy model for bird song classification.

  This model classifies the species of bird songs. It predicts multiple labels:
  whether a bird is detected, the species, genus, family, and order of the bird,
  and the nature of the background noise.

  Attributes:
    num_classes: Number of classes for each output head.
    encoder: A network (e.g., a 2D convolutional network) that takes
      spectrograms and returns feature vectors.
    taxonomy_loss_weight: Weight for taxonomic label losses. DEPRECATED!
    frontend: The frontend to use to generate features.
    hubert_feature_extractor: Optionally, a pre-trained frozen feature extractor
      trained in a self-supervised way. This option is mutually exclusive with
      frontend and is used for evaluation of self-supervised representations.
  """

  num_classes: dict[str, int]
  encoder: nn.Module
  taxonomy_loss_weight: float | None = None
  frontend: nn.Module | None = None
  hubert_feature_extractor: nn.Module | None = None

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      use_running_average: bool | None = None,
      mask: jnp.ndarray | None = None,
  ) -> dict[str, jnp.ndarray]:
    """Apply the taxonomy model.

    Args:
      inputs: Audio of shape `(batch size, time)`.
      train: Whether this is training. This affects Dropout behavior, and also
        affects BatchNorm behavior if 'use_running_average' is set to None.
      use_running_average: Optional, used to decide whether to use running
        statistics in BatchNorm (test mode), or the current batch's statistics
        (train mode). If not specified (or specified to None), default to 'not
        train'.
      mask: An optional mask of the inputs.
    Raises: ValueError if both `frontend` and `hubert_feature_extractor` are not
      None.

    Returns:
      Logits for each output head.
    """
    if self.frontend is not None and self.hubert_feature_extractor is not None:
      raise ValueError(
          "`frontend` and `hubert_feature_extractor` are mutually exclusive."
      )

    if use_running_average is None:
      use_running_average = not train
    kwargs = {} if mask is None else {"mask": mask}

    # Apply the frontend.
    if isinstance(self.frontend, layers.EarlyFeatureExtractor):
      # EarlyFeatureExtractor expects [B, T, C] inputs.
      x = self.frontend(inputs[:, :, jnp.newaxis], train=train)  # pylint: disable=not-callable
    elif self.frontend is not None:
      x = self.frontend(inputs, train=train)  # pylint: disable=not-callable
      if mask is not None:
        # Go from time steps to frames
        mask = frontend.frames_mask(mask, self.frontend.stride)
        # Add axes for broadcasting over frequencies and channels
        kwargs = {"mask": mask[..., jnp.newaxis, jnp.newaxis]}
    elif self.hubert_feature_extractor is not None:
      x = self.hubert_feature_extractor(inputs)  # pylint: disable=not-callable
    else:
      x = inputs
    frontend_outputs = x

    # Apply the encoder.
    while len(x.shape) < 4:
      # We may have shape (B, T), (B, T, D), or (B, W, H, D)
      x = x[..., jnp.newaxis]
    # Treat the spectrogram as a gray-scale image
    x = self.encoder(
        x, train=train, use_running_average=use_running_average, **kwargs
    )

    # Classify the encoder outputs and assemble outputs.
    model_outputs = {}
    model_outputs["embedding"] = x
    model_outputs["frontend"] = frontend_outputs
    for k, n in self.num_classes.items():
      model_outputs[k] = nn.Dense(n)(x)
    return model_outputs


class ConformerModel(nn.Module):
  """Conformer model."""

  num_conformer_blocks: int = 16
  features: int = 144
  num_heads: int = 4
  kernel_size: int = 15
  downsample: list[tuple[int, float]] = dataclasses.field(default_factory=list)

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      use_running_average: bool | None = None,
      mask: jnp.ndarray | None = None,
  ) -> jnp.ndarray:
    # Subsample from (x, 160) to (x // 4, 40)
    x = conformer.ConvolutionalSubsampling(features=self.features)(
        inputs, train=train
    )
    # Apply conformer blocks
    x = conformer.Conformer(
        model_dims=self.features,
        atten_num_heads=self.num_heads,
        num_blocks=self.num_conformer_blocks,
        kernel_size=self.kernel_size,
        downsample=self.downsample,
        dropout_prob=0.1,
    )(
        x,
        train=train,
        use_running_average=use_running_average,
        return_intermediate_list=False,
    )

    # To get a global embedding we now just pool
    return jnp.mean(x, axis=-2)
