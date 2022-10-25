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

"""NOisy TEacher-student with Laplacian Adjustment (NOTELA), our method."""

import functools
from typing import Dict, Tuple, Type

from absl import logging
from chirp.projects.sfda import adapt
from chirp.projects.sfda import losses
from chirp.projects.sfda import method_utils
from chirp.projects.sfda import model_utils
from clu import metrics as clu_metrics
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


class NOTELA(adapt.SFDAMethod):
  """NOisy TEacher-student with Laplacian Adjustment (NOTELA), our method.

  It builds upon Dropout Student by incorporating a Laplacian regularization
  to the teacher step. NOTELA works in two different modes:
    - offline mode: Pseudo-labels are computed only once every epoch (before the
    epoch starts).
    - online mode: (Coming in next release) We track a memory of the dataset's
      extracted features and probabilities. Pseudo-labels are computed on-the-go
      by comparing samples from the current batch to the features/probabilities
      in memory.
  In both cases, the student-step remains to match the pseudo-labels using a
  noisy (dropout) model forward.
  """

  @staticmethod
  def compute_nearest_neighbors(
      batch_feature: jnp.ndarray,
      dataset_feature: jnp.ndarray,
      knn: int,
  ) -> jnp.ndarray:
    """Compute batch_feature's nearest-neighbors among dataset_feature.

    Args:
      batch_feature: The features for the provided batch of data, shape
        [batch_size, feature_dim]
      dataset_feature: The features for the whole dataset, shape [dataset_size,
        feature_dim]
      knn: The number of nearest-neighbors to use.

    Returns:
      The batch's nearest-neighbors affinity matrix of shape
        [batch_size, dataset_size], where position (i, j) indicates whether
        dataset_feature[j] belongs to batch_feature[i]'s nearest-neighbors.

    Raises:
      ValueError: If batch_feature and dataset_feature don't have the same
        number of dimensions, or if their feature dimension don't match.
    """
    batch_shape = batch_feature.shape
    dataset_shape = dataset_feature.shape

    if batch_feature.ndim != dataset_feature.ndim or (batch_shape[-1] !=
                                                      dataset_shape[-1]):
      raise ValueError(
          "Batch features and dataset features' shapes are not consistent."
          f"Currently batch_feature: {batch_shape} and dataset_feature: {dataset_shape}"
      )

    # Compute the nearest-neighbors
    pairwise_distances = method_utils.jax_cdist(
        batch_feature, dataset_feature)  # [batch_size, dataset_size]
    neighbors = min(dataset_shape[0], knn)
    col_indexes = jax.lax.top_k(-pairwise_distances,
                                neighbors)[1][:,
                                              1:]  # [batch_size, neighbors-1]
    col_indexes = col_indexes.flatten()  # [batch_size * neighbors-1]
    row_indexes = jnp.repeat(np.arange(batch_shape[0]),
                             neighbors - 1)  # [1, ..., 1, 2, ...]
    # TODO(mboudiaf): Add option for sparse storage.
    nn_matrix = jnp.zeros((batch_shape[0], dataset_shape[0]), dtype=jnp.uint8)
    nn_matrix = nn_matrix.at[row_indexes, col_indexes].set(1)
    return nn_matrix

  @staticmethod
  def teacher_step(batch_proba: jnp.ndarray,
                   dataset_proba: jnp.ndarray,
                   nn_matrix: jnp.ndarray,
                   lambda_: float,
                   alpha: float = 1.0,
                   eps: float = 1e-8) -> jnp.ndarray:
    """Computes the pseudo-labels (teacher-step) following Eq.(3) in the paper.

    Args:
      batch_proba: The model's probabilities on the current batch of data.
        Expected shape [batch_size, proba_dim]
      dataset_proba: The model's probabilities on the rest of the dataset.
        Expected shape [dataset_size, proba_dim]
      nn_matrix: The affinity between the points in the current batch
        (associated to `batch_proba`) and the remaining of the points
        (associated to `dataset_proba`), of shape [batch_size, dataset_size].
        Specifically, position [i,j] informs if point j belongs to i's
        nearest-neighbors.
      lambda_: Weight controlling the Laplacian regularization.
      alpha: Weight controlling the Softness regularization
      eps: For numerical stability.

    Returns:
      The soft pseudo-labels for the current batch of data, shape
        [batch_size, proba_dim]
    """
    denominator = nn_matrix.sum(axis=-1, keepdims=True)
    pseudo_label = batch_proba**(1 / alpha) * jnp.exp(
        (lambda_ / alpha) * (nn_matrix @ dataset_proba) /
        (denominator + eps))  # [*, batch_size, proba_dim]
    pseudo_label /= (pseudo_label.sum(axis=-1, keepdims=True) + eps)
    return pseudo_label

  def before_run(self, key: jax.random.PRNGKeyArray,
                 model_bundle: model_utils.ModelBundle,
                 adaptation_state: adapt.AdaptationState,
                 adaptation_dataset: tf.data.Dataset, modality: adapt.Modality,
                 multi_label: bool, **method_kwargs) -> adapt.AdaptationState:
    """Initialize the memories when using NOTELA's online mode.

    Args:
      key: The jax random key used for random operations in this epoch.
      model_bundle: The ModelBundle used for adaptation.
      adaptation_state: The current state of adaptation.
      adaptation_dataset: The dataset used for adaptation.
      modality: The current modality.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Additional method-specific kwargs.

    Returns:
      An updated version of adaptation_state, where method_state contains
        all initialized memories.

    Raises:
      NotImplementedError: In the case the 'online mode' is activated.
    """
    if method_kwargs["online_pl_updates"]:
      raise NotImplementedError("Coming in the next release.")
    return adaptation_state

  def before_epoch(self, key: jax.random.PRNGKeyArray,
                   model_bundle: model_utils.ModelBundle,
                   adaptation_state: adapt.AdaptationState,
                   adaptation_dataset: tf.data.Dataset,
                   modality: adapt.Modality, multi_label: bool,
                   **method_kwargs) -> adapt.AdaptationState:
    """In 'offline mode', compute the pseudo-labels and store them in memory.

    If 'offline mode' is not activated, nothing needs to be done at that stage,
    as pseudo-labels will be computed on-the-go.

    Args:
      key: The jax random key used for random operations in this epoch.
      model_bundle: The ModelBundle used for adaptation.
      adaptation_state: The current state of adaptation.
      adaptation_dataset: The dataset used for adaptation.
      modality: The current modality.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Additional method-specific kwargs.

    Returns:
      The adaptation state, with a potentially updated 'method_state' attribute.
    """
    if not method_kwargs["online_pl_updates"]:

      logging.info("Preparing pseudo-labels...")

      # Extract embeddings and model's probabilities.
      forward_result = method_utils.forward_dataset(
          dataset=adaptation_dataset,
          adaptation_state=adaptation_state,
          model_bundle=model_bundle,
          modality=modality,
          multi_label=multi_label,
          use_batch_statistics=method_kwargs["update_bn_statistics"])

      # Compute pseudo-labels that will be used during the next epoch of
      # adaptation.
      pseudo_label = self.compute_pseudo_label(
          batch_feature=forward_result["embedding"],
          dataset_feature=forward_result["embedding"],
          batch_proba=forward_result["proba"],
          dataset_proba=forward_result["proba"],
          multi_label=multi_label,
          knn=method_kwargs["knn"],
          lambda_=method_kwargs["lambda_"],
          alpha=method_kwargs["alpha"])

      # method_state will act as a memory, from which pseudo-labels will be
      # grabbed on-the-go over the next epoch of adaptation.
      sample_ids = forward_result["id"]
      method_state = {
          "pseudo_label": pseudo_label,
          "id2index": {sample_ids[i]: i for i in range(len(sample_ids))},
      }
      adaptation_state = adaptation_state.replace(method_state=method_state)
    return adaptation_state

  def before_iter(
      self, key: jax.random.PRNGKeyArray, batch: Dict[str, np.ndarray],
      adaptation_state: adapt.AdaptationState,
      model_bundle: model_utils.ModelBundle, modality: adapt.Modality,
      multi_label: bool,
      **method_kwargs) -> Tuple[adapt.AdaptationState, Dict[str, jnp.ndarray]]:
    """Grab or compute the pseudo-labels for the current batch.

    In 'offline mode', grabs the pre-computed pseudo-labels from method_state's
    memory.

    Args:
      key: The jax random key used for random operations.
      batch: The current batch of data.
      adaptation_state: The current state of adaptation.
      model_bundle: The ModelBundle used for adaptation.
      modality: The current modality.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Additional method-specific kwarg

    Returns:
      If using offline mode, the untouched adaptation_state. Otherwise,an
        updated version in which the method_state's memories have been
        updated
      A dictionary containing the pseudo-labels to use for the iteration.

    Raises:
      NotImplementedError: If the online mode of NOTELA is activated.
    """

    if method_kwargs["online_pl_updates"]:
      raise NotImplementedError("Coming in the next release.")
    else:
      # In the offline version, we simply grab the pseudo-labels that were
      # computed before the epoch.
      method_state = flax_utils.unreplicate(adaptation_state.method_state)
      id2index = method_state["id2index"]
      batch_indexes = np.array(
          [id2index[x] for x in flax_utils.unreplicate(batch["tfds_id"])])
      pseudo_label = method_state["pseudo_label"][batch_indexes]
    return adaptation_state, {
        "pseudo_label": flax_utils.replicate(pseudo_label)
    }

  def compute_pseudo_label(
      self,
      batch_feature: jnp.ndarray,
      dataset_feature: jnp.ndarray,
      batch_proba: jnp.ndarray,
      dataset_proba: jnp.ndarray,
      multi_label: bool,
      knn: int,
      lambda_: float,
      alpha: float,
  ) -> jnp.ndarray:
    """The pipeline for computing NOTELA's pseudo labels.

    First, we compute the nearest neighbors of each point in batch_feature
    to each point in dataset_feature. Then, we compute the pseudo-labels
    using Eq. (3) from the paper.

    Args:
      batch_feature: The features for the provided batch of data, shape
        [batch_size, feature_dim]
      dataset_feature: The features for the whole dataset, shape [dataset_size,
        feature_dim]
      batch_proba: The model's proba for the current batch of data, shape
        [batch_size, num_classes]
      dataset_proba: The model's proba for the whole dataset, shape
        [dataset_size, num_classes]
      multi_label: Whether this is a multi-label problem.
      knn: The number of nearest-neighbors use to compute the affinity matrix.
      lambda_: The weight controlling the Laplacian regularization.
      alpha: The weight controlling the softness regularization.

    Returns:
      The nearest-neighbor matrix used to compute the pseudo-labels.
      The pseudo-labels for the provided batch of data, shape
        [batch_size, num_classes].
    """
    # Start by computing the affinity matrix
    nn_matrix = self.compute_nearest_neighbors(
        batch_feature=batch_feature, dataset_feature=dataset_feature, knn=knn)

    # Prepare the teacher function.
    teacher_step_fn = functools.partial(
        self.teacher_step, nn_matrix=nn_matrix, lambda_=lambda_, alpha=alpha)

    if multi_label:
      # In the multi-label scnenario, we're solving `num_classes` independent
      # binary problems. Therefore, the class dimension can be treated as a
      # batch dimension, and the 'probability dimension' is actually 2.
      def reshape_binary_probabilities(proba):
        proba = proba.T
        return jnp.stack([1 - proba, proba], axis=-1)

      dataset_proba = reshape_binary_probabilities(
          dataset_proba)  # [num_classes, dataset_size, 2]
      batch_proba = reshape_binary_probabilities(
          batch_proba)  # [num_classes, batch_size, 2]
      pseudo_label = []
      for classwise_batch_proba, classwise_dataset_proba in zip(
          batch_proba, dataset_proba):
        pseudo_label.append(
            teacher_step_fn(
                batch_proba=classwise_batch_proba,
                dataset_proba=classwise_dataset_proba))
      pseudo_label = jnp.stack(pseudo_label)  # [num_classes, batch_size, 2]
      # We select the 'positive' probability
      pseudo_label = pseudo_label[..., -1].T  # [batch_size, num_classes]
    else:
      pseudo_label = teacher_step_fn(
          batch_proba=batch_proba,
          dataset_proba=dataset_proba)  # [batch_size, num_classes]

    return jax.lax.stop_gradient(pseudo_label)

  def get_adaptation_metrics(self, supervised: bool, multi_label: bool,
                             **method_kwargs) -> Type[clu_metrics.Collection]:
    """Obtain metrics that will be monitored during adaptation.

    In NOTELA, the loss minimized w.r.t. the network is a simple cross-entropy
    between the model's (noisy) outputs and the pseudo-labels.

    Args:
      supervised: Whether the problem is supervised. Only used to know if we can
        track supervised metrics, such as accuracy.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Method's kwargs.

    Returns:
      A collection of metrics.
    """
    metrics_dict = vars(
        adapt.get_common_metrics(
            supervised=supervised, multi_label=multi_label))["__annotations__"]

    def single_label_loss_fn(probabilities, pseudo_label, **_):
      pl_xent = losses.label_xent(
          probabilities=probabilities, label=pseudo_label)
      return pl_xent

    def multi_label_loss_fn(probabilities: jnp.ndarray,
                            pseudo_label: jnp.ndarray, label_mask: jnp.ndarray,
                            **_):
      pl_xent = losses.label_binary_xent(
          probabilities=probabilities,
          label=pseudo_label,
          label_mask=label_mask,
      )
      return pl_xent

    loss_fn = multi_label_loss_fn if multi_label else single_label_loss_fn
    metrics_dict["main_loss"] = clu_metrics.Average.from_fun(loss_fn)
    return clu_metrics.Collection.create(**metrics_dict)
