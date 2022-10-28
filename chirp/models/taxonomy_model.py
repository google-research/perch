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

"""Taxonomy model."""
import dataclasses
from typing import Dict, List, Optional, Tuple

from chirp.models import conformer
from chirp.models import frontend
from chirp.models import layers
from chirp.models import hubert
import flax
from flax import linen as nn
from jax import numpy as jnp


@flax.struct.dataclass
class ModelOutputs:
  embedding: jnp.ndarray
  label: jnp.ndarray
  genus: Optional[jnp.ndarray] = None
  family: Optional[jnp.ndarray] = None
  order: Optional[jnp.ndarray] = None


class TaxonomyModel(nn.Module):
  """Taxonomy model for bird song classification.

  This model classifies the species of bird songs. It predicts multiple labels:
  whether a bird is detected, the species, genus, family, and order of the bird,
  and the nature of the background noise.

  Attributes:
    num_classes: Number of classes for each output head.
    encoder: A network (e.g., a 2D convolutional network) that takes
      spectrograms and returns feature vectors.
    taxonomy_loss_weight: Weight for taxonomic label losses.
    frontend: The frontend to use to generate features.
    hubert_feature_extractor: Optionally, a pre-trained frozen feature extractor
      trained in a self-superivsed way. This option is mutually exclusive with
      frontend and is used for evaluation of self-superviesd representations.
  """
  num_classes: Dict[str, int]
  encoder: nn.Module
  taxonomy_loss_weight: float
  frontend: Optional[nn.Module] = None
  hubert_feature_extractor: Optional[nn.Module] = None

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               train: bool,
               use_running_average: Optional[bool] = None,
               mask: Optional[jnp.ndarray] = None) -> ModelOutputs:
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
      raise ValueError("`frontend` and `hubert_feature_extractor` are mutually "
                       "exclusive.")

    if use_running_average is None:
      use_running_average = not train
    kwargs = {} if mask is None else {"mask": mask}
    if self.frontend is not None:
      x = self.frontend(inputs)
      if mask is not None:
        # Go from time steps to frames
        mask = frontend.frames_mask(mask, self.frontend.stride)
        # Add axes for broadcasting over frequencies and channels
        kwargs = {"mask": mask[..., jnp.newaxis, jnp.newaxis]}
    elif self.hubert_feature_extractor is not None:
      x = self.hubert_feature_extractor(inputs)
    else:
      x = inputs
    if isinstance(self.encoder, conformer.Conformer):
      x = self.encoder(x, train=train, use_running_average=use_running_average)
      # Silly baseline: average over the time dimension.
      x = jnp.mean(x, axis=1)
    else:
      # Treat the spectrogram as a gray-scale image
      x = self.encoder(
          x[..., jnp.newaxis],
          train=train,
          use_running_average=use_running_average,
          **kwargs)

    model_outputs = {}
    model_outputs["embedding"] = x
    for k, n in self.num_classes.items():
      if self.taxonomy_loss_weight == 0.0 and k != "label":
        continue
      model_outputs[k] = nn.Dense(n)(x)
    return ModelOutputs(**model_outputs)


class ConformerModel(nn.Module):
  """Conformer model."""
  num_conformer_blocks: int = 16
  features: int = 144
  num_heads: int = 4
  kernel_size: int = 15
  downsample: List[Tuple[int, float]] = dataclasses.field(default_factory=list)

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               train: bool,
               use_running_average: Optional[bool] = None,
               mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    # Apply frontend
    dim = 512
    conv_layer_tuples = ((dim, 20, 10), (dim, 3, 2), (dim, 3, 2), (dim, 3, 2),
                         (dim, 3, 2), (dim, 2, 2), (160, 2, 2))
    x = layers.EarlyFeatureExtractor(conv_layer_tuples)(inputs, train=train)

    # At this point, think of channels as frequencies and add single channel
    x = x[..., jnp.newaxis]

    # Subsample from (x, 160) to (x // 4, 40)
    x = conformer.ConvolutionalSubsampling(features=self.features)(
        x, train=train)

    # Apply conformer blocks
    x = conformer.Conformer(
        model_dims=self.features,
        atten_num_heads=self.num_heads,
        num_blocks=self.num_conformer_blocks,
        kernel_size=self.kernel_size,
        downsample=self.downsample,
        dropout_prob=0.1)(
            x,
            train=train,
            use_running_average=use_running_average,
            return_intermediate_list=False)

    # To get a global embedding we now just pool
    return jnp.mean(x, axis=-2)
