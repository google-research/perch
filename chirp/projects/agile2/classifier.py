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

"""Functions for training and applying a linear classifier."""

from typing import Any

from chirp.models import metrics
from chirp.projects.agile2 import classifier_data
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm


def hinge_loss(pred: jax.Array, y: jax.Array, w: jax.Array) -> jax.Array:
  """Weighted SVM hinge loss."""
  # Convert multihot to +/- 1 labels.
  y = 2 * y - 1
  return w * jnp.maximum(0, 1 - y * pred)


def bce_loss(pred: jax.Array, y: jax.Array, w: jax.Array) -> jax.Array:
  return w * optax.losses.sigmoid_binary_cross_entropy(pred, y)


def infer(params, embeddings: jax.Array | np.ndarray):
  """Apply the model to embeddings."""
  return jnp.dot(embeddings, params['beta']) + params['beta_bias']


def forward(
    params,
    batch,
    weak_neg_weight: float,
    l2_mu: float,
    loss_name: str = 'hinge',
) -> jax.Array:
  """Forward pass for classifier training."""
  embeddings = batch.embedding
  pred = infer(params, embeddings)
  weights = (
      batch.is_labeled_mask + (1.0 - batch.is_labeled_mask) * weak_neg_weight
  )
  # Loss shape is [B, C]
  if loss_name == 'hinge':
    loss = hinge_loss(pred=pred, y=batch.multihot, w=weights).sum()
  elif loss_name == 'bce':
    loss = bce_loss(pred=pred, y=batch.multihot, w=weights).sum()
  else:
    raise ValueError(f'Unknown loss name: {loss_name}')
  l2_reg = jnp.dot(params['beta'].T, params['beta']).mean()
  loss = loss + l2_mu * l2_reg
  return loss.mean()


def eval_classifier(
    params: Any,
    data_manager: classifier_data.DataManager,
    eval_ids: np.ndarray,
) -> dict[str, float]:
  """Evaluate a classifier on a set of examples."""
  iter_ = data_manager.batched_example_iterator(
      eval_ids, add_weak_negatives=False, repeat=False
  )
  # The embedding ids may be shuffled by the iterator, so we will track the ids
  # of the examples we are evaluating.
  got_ids = []
  pred_logits = []
  true_labels = []
  for batch in iter_:
    pred_logits.append(infer(params, batch.embedding))
    true_labels.append(batch.multihot)
    got_ids.append(batch.idx)
  pred_logits = np.concatenate(pred_logits, axis=0)
  true_labels = np.concatenate(true_labels, axis=0)
  got_ids = np.concatenate(got_ids, axis=0)

  # Compute the top1 accuracy on examples with at least one label.
  labeled_locs = np.where(true_labels.sum(axis=1) > 0)
  top_preds = np.argmax(pred_logits, axis=1)
  top1 = true_labels[np.arange(top_preds.shape[0]), top_preds]
  top1 = top1[labeled_locs].mean()

  rocs = metrics.roc_auc(
      logits=pred_logits, labels=true_labels, sample_threshold=1
  )
  cmaps = metrics.cmap(
      logits=pred_logits, labels=true_labels, sample_threshold=1
  )
  return {
      'top1_acc': top1,
      'roc_auc': rocs['macro'],
      'roc_auc_individual': rocs['individual'],
      'cmap': cmaps['macro'],
      'cmap_individual': cmaps['individual'],
      'eval_ids': got_ids,
      'eval_preds': pred_logits,
      'eval_labels': true_labels,
  }


def train_linear_classifier(
    data_manager: classifier_data.DataManager,
    learning_rate: float,
    weak_neg_weight: float,
    l2_mu: float,
    num_train_steps: int,
    loss_name: str = 'hinge',
):
  """Train a linear classifier."""
  optimizer = optax.adam(learning_rate=learning_rate)
  embedding_dim = data_manager.db.embedding_dimension()
  num_classes = len(data_manager.target_labels)
  params = {
      'beta': jnp.zeros((embedding_dim, num_classes)),
      'beta_bias': jnp.zeros((num_classes,)),
  }
  opt_state = optimizer.init(params)

  def update(params, batch, opt_state, **kwargs) -> jax.Array:
    loss, grads = jax.value_and_grad(forward)(params, batch, **kwargs)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

  train_ids, eval_ids = data_manager.get_train_test_split()
  iter_ = data_manager.batched_example_iterator(
      train_ids, add_weak_negatives=True, repeat=True
  )

  progress = tqdm.tqdm(enumerate(iter_), total=num_train_steps)
  for step, batch in enumerate(iter_):
    if step >= num_train_steps:
      break
    loss, params, opt_state = update(
        params,
        batch,
        opt_state,
        weak_neg_weight=weak_neg_weight,
        l2_mu=l2_mu,
        loss_name=loss_name,
    )
    progress.update()
    progress.set_description(f'Loss {loss:.8f}')

  eval_scores = eval_classifier(params, data_manager, eval_ids)
  return params, eval_scores
