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

from chirp.models import conformer
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
    frontend: The frontend to use to generate features.
    encoder: A network (e.g., a 2D convolutional network) that takes
      spectrograms and returns feature vectors.
    taxonomy_loss_weight: Weight for taxonomic label losses.
  """
  num_classes: Dict[str, int]
  frontend: nn.Module
  encoder: nn.Module
  taxonomy_loss_weight: float

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> ModelOutputs:
    """Apply the taxonomy model.

    Args:
      inputs: Audio of shape `(batch size, time)`.
      train: Whether this is training (affects batch norm and dropout).

    Returns:
      Logits for each output head.
    """
    x = self.frontend(inputs)
    if isinstance(self.encoder, conformer.Conformer):
      x = self.encoder(x, train=train)
      # Silly baseline: average over the time dimension.
      x = jnp.mean(x, axis=1)
    else:
      # Treat the spectrogram as a gray-scale image
      x = self.encoder(x[..., jnp.newaxis], train=train)

    model_outputs = {}
    model_outputs["embedding"] = x
    for k, n in self.num_classes.items():
      if self.taxonomy_loss_weight == 0.0 and k != "label":
        continue
      model_outputs[k] = nn.Dense(n)(x)
    return ModelOutputs(**model_outputs)
