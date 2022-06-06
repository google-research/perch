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
from typing import Dict, Optional

from chirp import audio_utils
from chirp import signal
from chirp.models import efficientnet
import flax
from flax import linen as nn
from jax import numpy as jnp
from ml_collections import config_dict


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
    encoder: A network (e.g., a 2D convolutional network) that takes
      spectrograms and returns feature vectors.
    taxonomy_loss_weight: Weight for taxonomic label losses.
    bandwidth: The number of frequencies in each band.
    band_stride: The number of frequencies between each band.
    num_classes: Number of classes for each output head.
  """
  num_classes: Dict[str, int]
  melspec_config: config_dict.ConfigDict
  random_low_pass: bool
  robust_normalization: bool
  bandwidth: int
  taxonomy_loss_weight: float
  band_stride: Optional[int] = None
  encoder: nn.Module = efficientnet.EfficientNet(
      efficientnet.EfficientNetModel.B1)

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> ModelOutputs:
    """Apply the taxonomy model.

    Args:
      inputs: Audio of shape `(batch size, time)`.
      train: Whether this is training (affects batch norm and dropout).

    Returns:
      Logits for each output head.
    """
    # TODO(bartvm): Factor out frontend
    x = audio_utils.compute_melspec(inputs, **self.melspec_config)
    if train and self.random_low_pass:
      x = audio_utils.random_low_pass_filter(self.make_rng("low_pass"), x)
    if self.robust_normalization:
      # TODO(bartvm): Understand instability and compare against old version
      med = jnp.median(x, axis=1, keepdims=True)
      mad = jnp.median(jnp.abs(x - med), axis=1, keepdims=True)
      x = (x - med) / (mad + 1e-3)
    if self.bandwidth <= 0:
      x = x[..., jnp.newaxis]
    else:
      band_stride = self.band_stride or self.bandwidth
      x = signal.frame(x, self.bandwidth, band_stride)
      x = jnp.swapaxes(x, 2, 3)

    # Apply the encoder
    x = self.encoder(x, train=train)

    model_outputs = {}
    model_outputs["embedding"] = x
    for k, n in self.num_classes.items():
      if self.taxonomy_loss_weight == 0.0 and k != "label":
        continue
      model_outputs[k] = nn.Dense(n)(x)
    return ModelOutputs(**model_outputs)
