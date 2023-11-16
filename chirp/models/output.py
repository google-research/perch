# coding=utf-8
# Copyright 2023 The Perch Authors.
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

"""Model outputs."""
import dataclasses
from typing import Protocol, runtime_checkable

import flax
from jax import numpy as jnp


@flax.struct.dataclass
class EmbeddingOutput:
  embedding: jnp.ndarray


@flax.struct.dataclass
class ClassifierOutput(EmbeddingOutput):
  label: jnp.ndarray


@flax.struct.dataclass
class TaxonomyOutput(ClassifierOutput):
  genus: jnp.ndarray
  family: jnp.ndarray
  order: jnp.ndarray


@runtime_checkable
class AnyOutput(Protocol):
  """Any output must be a dataclass."""

  __dataclass_fields__: dict[str, dataclasses.Field]  # pylint: disable=g-bare-generic


@runtime_checkable
@dataclasses.dataclass
class TaxonomicOutput(Protocol):
  label: jnp.ndarray
  genus: jnp.ndarray
  family: jnp.ndarray
  order: jnp.ndarray


def output_head_logits(output, output_head_metadatas) -> dict[str, jnp.ndarray]:
  return {
      f'{md.key}_logits': output[md.key]
      for md in output_head_metadatas
      if md.key in output
  }


def logits(output) -> dict[str, jnp.ndarray]:
  return {
      f'{key}_logits': getattr(output, key)
      for key in ('label', 'genus', 'family', 'order')
      if hasattr(output, key)
  }
