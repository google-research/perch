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

"""Code for validating using one-shot retrieval."""
import statistics
from typing import Callable

from chirp.models import metrics
from grain import python as pygrain
import jax
from jax import numpy as jnp
from jax import random


def embed(
    data_loader: pygrain.DataLoader, embed_fn: Callable[[jax.Array], jax.Array]
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Embeds a dataset and returns the embeddings and labels.

  Args:
    data_loader: Data loader for the dataset to embed.
    embed_fn: Function used for embedding the data.

  Returns:
    A tuple containing the embedded dataset and the labels.
  """
  embeddings, labels = [], []
  for batch in data_loader:
    # Each element in the dataset is a recording that was turned into a batch
    # of windows.
    embeddings.append(embed_fn(batch["windows"]))
    labels.append(batch["label"])
  return jnp.concatenate(embeddings, axis=0), jnp.concatenate(labels, axis=0)


def sample_queries(
    key: jax.Array, labels: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Samples a one-shot retrieval query for each species.

  Args:
    key: The random sampling key.
    labels: The dataset's labels with shape [num_examples, num_species].

  Returns:
    A tuple containing the selected example indices for each species (shape
    [num_species]) and a label mask (shape [num_examples, num_species]) to be
    used in computing metrics.
  """
  choose_one_per_species = jax.vmap(
      lambda k, l: jax.random.choice(k, a=len(l), p=l / l.sum())
  )

  chosen_indices = choose_one_per_species(
      random.split(key, labels.shape[1]), labels.T
  )
  label_mask = jnp.ones(shape=labels.shape, dtype=jnp.int32)
  label_mask = label_mask.at[chosen_indices, jnp.arange(labels.shape[1])].set(0)

  return chosen_indices, label_mask


def one_shot_metric(
    key: jax.Array, normalized_embeddings: jax.Array, labels: jax.Array
) -> float:
  """Computes the species-aggregated ROC-AUC.

  The metric is computed on a one-shot retrieval problem for each species and
  uses the cosine similarity to rank embeddings with respect to each retrieval
  query.

  Note that when this metric constructs a one-shot retrieval task, it
  effectively ignores all other labels. That is, when constructing a retrieval
  task for class A, an example with label set A and B can be chosen as the
  retrieval query (exemplar). Then, when an example with label B is returned it
  is considered a false positive.

  This function is also not aware of any "unknown" class. Consider removing
  all examples with this label entirely, because it is often not sensible to
  use examples with this label as retrieval queries, nor is it clear whether
  such examples are false positives when retrieved.

  Args:
    key: Random key controlling the sampling of one-shot retrieval queries  .
    normalized_embeddings: Search corpus of normalized embeddings.
    labels: Multi-hot labels. Only classes with 2 or more samples will be used,
      all others will be ignored. Note that the classes should probably not
      include an unknown class, because it makes little sense to try and do
      one-shot retrieval on those samples.

  Returns:
    The ROC-AUC, aggregated across species using the geometric average.
  """
  exemplar_indices, label_mask = sample_queries(key, labels)
  return float(
      metrics.roc_auc(
          logits=jnp.tensordot(
              normalized_embeddings,
              normalized_embeddings[exemplar_indices],
              [[-1], [-1]],
          ),
          labels=labels,
          label_mask=label_mask,
          sort_descending=True,
          # We need at least 2 samples for one-shot retrieval: One exemplar and
          # one to retrieve.
          sample_threshold=2,
      )["geometric"]
  )


def one_shot_validate(
    key: jax.Array,
    data_loader: pygrain.DataLoader,
    embed_fn: Callable[[jax.Array], jax.Array],
    num_samples: int,
) -> float:
  """Computes the validation metric.

  The metric is the species-aggregated ROC-AUC for one-shot retrieval, averaged
  over `num_samples` random samplings of the retrieval queries.

  Args:
    key: Random key controlling the sampling of one-shot retrieval queries.
    data_loader: Data loader for the dataset to embed.
    embed_fn: The embeddinf function used to embed audio.
    num_samples: The number of one-shot retrieval tasks to sample.

  Returns:
    The validation metric.
  """
  embeddings, labels = embed(data_loader, embed_fn)
  normalized_embeddings = embeddings / jnp.linalg.norm(
      embeddings, axis=-1, keepdims=True
  )

  roc_aucs = [
      one_shot_metric(random.fold_in(key, i), normalized_embeddings, labels)
      for i in range(num_samples)
  ]
  return statistics.mean(roc_aucs)
