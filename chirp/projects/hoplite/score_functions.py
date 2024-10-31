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

"""Score functions for Hoplite."""

from typing import Callable
import numpy as np


def get_score_fn(
    name: str, bias: float | None = None, target_score: float | None = None
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
  """Get a score function by name."""
  if name == 'dot':
    score_fn = numpy_dot
  elif name == 'cos':
    score_fn = numpy_cos
  elif name == 'neg_euclidean':
    score_fn = numpy_neg_euclidean
  else:
    raise ValueError('Unknown score function: ', name)

  if bias is not None:
    bias_fn = lambda x, y: score_fn(x, y) + bias
  else:
    bias_fn = score_fn

  if target_score is not None:
    # We want 'up is good', so take the negative absolute value.
    targeted_fn = lambda x, y: -np.abs(bias_fn(x, y) - target_score)
  else:
    targeted_fn = bias_fn

  return targeted_fn


def numpy_dot(data: np.ndarray, query: np.ndarray) -> np.ndarray:
  """Simple numpy dot product, which allows for multiple queries."""
  if len(query.shape) > 1:
    # Tensordot is a little faster than dot for multiple queries, but slower
    # than dot for single queries.
    return np.tensordot(data, query, axes=(-1, -1))
  return np.dot(data, query)


def numpy_cos(data: np.ndarray, query: np.ndarray) -> np.ndarray:
  """Simple numpy cosine similarity, allowing multiple queries."""
  data_norms = np.linalg.norm(data, axis=1)
  query_norms = np.linalg.norm(query, axis=-1)
  if len(query.shape) > 1:
    unit_data = data / data_norms[:, np.newaxis]
    unit_query = query / query_norms[:, np.newaxis]
    return np.tensordot(unit_data, unit_query, axes=(-1, -1))
  unit_data = data / data_norms
  unit_query = query / query_norms
  return np.dot(unit_data, unit_query)


def numpy_neg_euclidean(data: np.ndarray, query: np.ndarray) -> np.ndarray:
  """Negative L2 distance allowing multiple queries."""
  data_norms = np.linalg.norm(data, axis=-1)
  if len(query.shape) > 1:
    query_norms = np.linalg.norm(query, axis=-1)
    dot_products = np.tensordot(data, query, axes=(-1, -1))
    pairs = data_norms[:, np.newaxis] + query_norms[np.newaxis, :]
    return -pairs + 2 * dot_products

  query_norm = np.linalg.norm(query)
  dot_products = np.dot(data, query)
  return -data_norms + 2 * dot_products + query_norm


def get_jax_dot():
  """Create a composite jax dot product function."""
  # Jax is an optional dependency.
  # pylint: disable-next=g-import-not-at-top
  from jax import numpy as jnp

  def composite_dot(data, query):
    if len(query.shape) > 1:
      return jnp.tensordot(data, query, axes=(-1, -1))
    else:
      # Numpy is faster for simple matrix-vector products.
      return np.dot(data, query)

  return composite_dot
