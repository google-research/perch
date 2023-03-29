# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""Exploiting the Intrinsic Neighborhood Structure for SFDA."""

from absl import logging
from chirp.projects.sfda import adapt
from chirp.projects.sfda import losses
from chirp.projects.sfda import method_utils
from chirp.projects.sfda import model_utils
from clu import metrics as clu_metrics
import flax
import flax.jax_utils as flax_utils
import flax.linen as flax_linen
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


@flax.struct.dataclass
class NRCLoss(clu_metrics.Metric):
  """Computes NRC's loss for the standard single-label case."""

  probabilities_sum: jnp.array
  nn_loss_sum: jnp.array
  extended_nn_loss_sum: jnp.array
  n_samples: int

  @classmethod
  def from_model_output(
      cls,
      probabilities: jnp.ndarray,
      nn_probability: jnp.ndarray,
      extended_nn_probability: jnp.ndarray,
      nn_weight: jnp.ndarray,
      extended_nn_weight: jnp.ndarray,
      **_,
  ) -> "NRCLoss":
    """Computes the standard extended nearest-neighbors loss.

    Args:
      probabilities: Model's probability for the batch.
      nn_probability: Batch's nearest-neighbors' probability vectors.
      extended_nn_probability: Batch's extended nearest-neighbors' probability
        vectors.
      nn_weight: The weight used for each nearest-neighbor. Expected shape
        [batch_size, nn]
      extended_nn_weight: The weight used for each extended nearest-neighbor.
        Expected shape [1] (as the same weight is used for all extended
        neighbors).

    Returns:
      NRCLoss: An instance of NRCLoss.
    """

    nn_loss = -(
        nn_weight * (probabilities[:, None, :] * nn_probability).sum(axis=-1)
    ).sum(
        axis=-1
    )  # [batch_size]

    extended_nn_loss = -(
        extended_nn_weight
        * (probabilities[:, None, None, :] * extended_nn_probability).sum(
            axis=-1
        )
    ).sum(
        axis=1
    )  # [batch_size]
    probabilities_sum = probabilities.sum(axis=0)  # [num classes]

    return cls(
        probabilities_sum=probabilities_sum,
        nn_loss_sum=nn_loss.sum(),
        extended_nn_loss_sum=extended_nn_loss.sum(),
        n_samples=probabilities.shape[0],
    )

  def merge(self, other: "NRCLoss") -> "NRCLoss":
    return type(self)(
        probabilities_sum=self.probabilities_sum + other.probabilities_sum,
        nn_loss_sum=self.nn_loss_sum + other.nn_loss_sum,
        extended_nn_loss_sum=self.extended_nn_loss_sum
        + other.extended_nn_loss_sum,
        n_samples=self.n_samples + other.n_samples,
    )

  def compute(self):
    probabilities_marginal = self.probabilities_sum / self.n_samples
    # TODO(mboudiaf): fix the single-label case in the audio setting.
    marginal_entropy = losses.label_ent(
        probabilities=probabilities_marginal, label_mask=None
    )
    return (
        1 / self.n_samples * (self.nn_loss_sum + self.extended_nn_loss_sum)
        - marginal_entropy
    )


@flax.struct.dataclass
class NRCMultiLoss(clu_metrics.Metric):
  """Computes NRC's loss for the multi-label case."""

  probabilities_sum: jnp.array
  nn_loss_sum: jnp.array
  extended_nn_loss_sum: jnp.array
  label_mask: jnp.array
  n_samples: int

  @classmethod
  def from_model_output(
      cls,
      probabilities: jnp.ndarray,
      nn_probability: jnp.ndarray,
      extended_nn_probability: jnp.ndarray,
      nn_weight: jnp.ndarray,
      extended_nn_weight: jnp.ndarray,
      label_mask: jnp.ndarray,
      **_,
  ) -> "NRCMultiLoss":
    if label_mask is not None:
      # probabilities have not been masked but nn_probability has been, so we
      # pad the latter to bring it to the same dimensionality as the former.
      reference_mask = label_mask[0]
      _, num_nn, num_classes_used = nn_probability.shape
      nn_probability_flatter = nn_probability.reshape((-1, num_classes_used))
      batch_size = nn_probability_flatter.shape[0]
      num_classes_total = reference_mask.shape[0]
      padded_nn_prob = jnp.zeros((batch_size, num_classes_total))
      col_index = jnp.tile(
          jnp.nonzero(reference_mask, size=num_classes_used)[0], batch_size
      )
      row_index = jnp.repeat(jnp.arange(batch_size), num_classes_used)
      nn_probability_flatter = padded_nn_prob.at[(row_index, col_index)].set(
          nn_probability_flatter.flatten()
      )
      nn_probability = nn_probability_flatter.reshape(
          (-1, num_nn, num_classes_total)
      )

      _, num_nn, num_enn, num_classes_used = extended_nn_probability.shape
      enn_probability_flatter = extended_nn_probability.reshape(
          (-1, num_classes_used)
      )
      batch_size = enn_probability_flatter.shape[0]
      padded_enn_prob = jnp.zeros((batch_size, num_classes_total))
      col_index = jnp.tile(
          jnp.nonzero(reference_mask, size=num_classes_used)[0], batch_size
      )
      row_index = jnp.repeat(jnp.arange(batch_size), num_classes_used)
      enn_probability_flatter = padded_enn_prob.at[(row_index, col_index)].set(
          enn_probability_flatter.flatten()
      )
      extended_nn_probability = enn_probability_flatter.reshape(
          (-1, num_nn, num_enn, num_classes_total)
      )

    def dot_product(probability_a, probability_b):
      return probability_a * probability_b + (1 - probability_a) * (
          1 - probability_b
      )

    nn_loss = -(
        label_mask
        * (
            nn_weight[..., None]  # [batch_size, nn, 1]
            * (dot_product(probabilities[:, None, :], nn_probability))
        ).sum(axis=1)
    ).sum(-1) / label_mask.sum(
        -1
    )  # [batch_size]

    extended_nn_loss = -(
        label_mask
        * (  # pytype: disable=wrong-arg-types  # jax-ndarray
            extended_nn_weight
            * (
                dot_product(
                    probabilities[:, None, None, :], extended_nn_probability
                )
            )
        ).sum(axis=[1, 2])
    ).sum(-1) / label_mask.sum(
        -1
    )  # [batch_size]

    probabilities_sum = probabilities.sum(axis=0)  # [num classes]

    return cls(
        probabilities_sum=probabilities_sum,
        nn_loss_sum=nn_loss.sum(),
        extended_nn_loss_sum=extended_nn_loss.sum(),
        label_mask=label_mask,
        n_samples=probabilities.shape[0],
    )

  def merge(self, other: "NRCMultiLoss") -> "NRCMultiLoss":
    return type(self)(
        probabilities_sum=self.probabilities_sum + other.probabilities_sum,
        nn_loss_sum=self.nn_loss_sum + other.nn_loss_sum,
        extended_nn_loss_sum=self.extended_nn_loss_sum
        + other.extended_nn_loss_sum,
        n_samples=self.n_samples + other.n_samples,
        label_mask=other.label_mask,
    )

  def compute(self):
    probabilities_marginal = self.probabilities_sum / self.n_samples
    marginal_entropy = losses.label_binary_ent(
        probabilities=probabilities_marginal, label_mask=self.label_mask[0]
    )
    return (
        1 / self.n_samples * (self.nn_loss_sum + self.extended_nn_loss_sum)
        - marginal_entropy
    )


class NRC(adapt.SFDAMethod):
  """Exploiting the Intrinsic Neighborhood Structure for SFDA."""

  _CITATION = (
      "Yang, Shiqi, et al. 'Exploiting the intrinsic neighborhood structure"
      "for source-free domain adaptation.' Advances in Neural Information"
      "Processing Systems 34 (2021): 29393-29405."
  )

  @staticmethod
  def compute_nearest_neighbors(
      batch_feature: jnp.ndarray,
      dataset_feature: jnp.ndarray,
      nn: int,
      memory_efficient_computation: bool = False,
  ) -> jnp.ndarray:
    """Compute batch_feature's nearest-neighbors among dataset_feature.

    Args:
      batch_feature: The features for the provided batch of data, shape
        [batch_size, feature_dim]
      dataset_feature: The features for the whole dataset, shape [dataset_size,
        feature_dim]
      nn: The number of nearest-neighbors to use.
      memory_efficient_computation: whether to use a memory-efficient
        implementation.

    Returns:
      The indices of batch_feature's nn nearest-neighbors among
      dataset_feature. Shape [batch_size, nn]

    Raises:
      ValueError if batch_feature and dataset_feature's shape don't match.
    """
    batch_shape = batch_feature.shape
    dataset_shape = dataset_feature.shape

    if batch_feature.ndim != dataset_feature.ndim or (
        batch_shape[-1] != dataset_shape[-1]
    ):
      raise ValueError(
          "Batch features and dataset features' shapes are not consistent."
          f"Currently batch_feature: {batch_shape} and dataset_feature:"
          f"{dataset_shape}"
      )

    # Compute the nearest-neighbors
    neighbors = min(dataset_shape[0], nn + 1)
    if memory_efficient_computation:
      # We loop over samples in the current batch to avoid storing a
      # batch_size x dataset_size float array. That slows down computation, but
      # reduces memory footprint, which becomes the bottleneck for large
      # datasets.
      nn_indices = []
      for sample_feature in batch_feature:
        pairwise_distances = method_utils.jax_cdist(
            jnp.expand_dims(sample_feature, 0), dataset_feature
        )  # [1, dataset_size]
        nn_indices.append(
            jax.lax.top_k(-pairwise_distances, neighbors)[1][:, 1:]
        )  # [1, neighbors]
      nn_indices = jnp.concatenate(
          nn_indices, axis=0
      )  # [batch_size, neighbors]
    else:
      pairwise_distances = method_utils.jax_cdist(
          batch_feature, dataset_feature
      )
      nn_indices = jax.lax.top_k(-pairwise_distances, neighbors)[1][
          :, 1:
      ]  # [batch_size, neighbors]

    return nn_indices

  def before_run(
      self,
      key: jax.random.PRNGKeyArray,
      model_bundle: model_utils.ModelBundle,
      adaptation_state: adapt.AdaptationState,
      adaptation_dataset: tf.data.Dataset,
      modality: adapt.Modality,
      multi_label: bool,
      **method_kwargs,
  ) -> adapt.AdaptationState:
    """Initialize the probability and feature banks.

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
      all initialized banks.
    """
    logging.info("Initializing banks...")

    # Extract embeddings and model's probabilities.
    forward_result = method_utils.forward_dataset(
        dataset=adaptation_dataset,
        adaptation_state=adaptation_state,
        model_bundle=model_bundle,
        modality=modality,
        multi_label=multi_label,
        use_batch_statistics=method_kwargs["update_bn_statistics"],
    )

    # Store everything in the method_state dictionnary.
    ids = forward_result["id"]
    method_state = {
        "dataset_feature": forward_result["embedding"],
        "dataset_probability": forward_result["proba"],
        "id2index": {ids[i]: i for i in range(len(ids))},
    }
    adaptation_state = adaptation_state.replace(method_state=method_state)
    return adaptation_state

  def before_iter(
      self,
      key: jax.random.PRNGKeyArray,
      batch: dict[str, np.ndarray],
      adaptation_state: adapt.AdaptationState,
      model_bundle: model_utils.ModelBundle,
      modality: adapt.Modality,
      multi_label: bool,
      **method_kwargs,
  ) -> tuple[adapt.AdaptationState, dict[str, jnp.ndarray]]:
    """Compute the (extended-)nearest-neighbors probability and weights.

    NRC relies on aligning model's probabilities with 'pseudo-labels'
    computed from the (extended) nearest-neighbors. Here, we compute those
    pseudo-labels, and their associated `weights` (i.e. 1 for reciprocal
    nearest-neighbors, and 'base_affinity' for the others).

    Args:
      key: The jax random key used for random operations.
      batch: The current batch of data.
      adaptation_state: The current state of adaptation.
      model_bundle: The ModelBundle used for adaptation.
      modality: The current modality.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Additional method-specific kwarg

    Returns:
      A dictionary containing direct and extended nearest-neighbors's
      probability vectors and weights, for each sample in the batch.
    """
    method_state = flax_utils.unreplicate(adaptation_state.method_state)
    id2index = method_state["id2index"]
    batch_indices = np.array(
        [id2index[x] for x in flax_utils.unreplicate(batch["tfds_id"])]
    )
    reference_label_mask = method_utils.get_label_mask(batch)

    # Obtain the model's output for the current batch.
    forward_step = self.cache_get_forward_step(
        model_bundle.model, modality, method_kwargs["update_bn_statistics"]
    )
    model_outputs = forward_step(  # pytype: disable=wrong-arg-types  # jax-ndarray
        adapt.keep_jax_types(batch),
        adaptation_state.model_state,
        adaptation_state.model_params,
        None,
    )
    model_outputs = flax_utils.unreplicate(model_outputs)
    model_outputs = method_utils.maybe_restrict_labels(
        model_outputs, reference_label_mask, adaptation_state
    )

    logit2proba = flax_linen.sigmoid if multi_label else flax_linen.softmax

    # Compute nearest-neighbors and extended nearest-neighbors indices.
    nn_indices = self.compute_nearest_neighbors(
        batch_feature=model_outputs.embedding,
        dataset_feature=method_state["dataset_feature"],
        nn=method_kwargs["nn"],
    )  # [batch_size, nn]
    extended_nn_indices = jnp.stack(
        [  # pylint: disable=g-complex-comprehension
            self.compute_nearest_neighbors(
                batch_feature=method_state["dataset_feature"][
                    sample_nn_indices
                ],
                dataset_feature=method_state["dataset_feature"],
                nn=method_kwargs["extended_nn"],
            )
            for sample_nn_indices in nn_indices  # [nn, extended_nn]
        ],
        axis=0,
    )  # [batch_size, nn, extended_nn]

    # Get nearest-neighbors and extended nearest-neighbors' probability.
    nn_probability = method_state["dataset_probability"][
        nn_indices
    ]  # [batch_size, nn, num_classes]
    extended_nn_probability = method_state["dataset_probability"][
        extended_nn_indices
    ]  # [batch_size, nn, extended_nn, num_classes]

    # Compute weights for nearest-neighbors and extended nearest-neighbors.
    # Those indicate the importance of each (extended) nearest-neighbor in
    # the loss.
    match = (extended_nn_indices == batch_indices[:, None, None]).sum(
        -1
    )  # [batch_size, nn]
    assert match.ndim == 2

    nn_weight = jnp.where(
        match > 0, match, method_kwargs["base_affinity"]
    )  # [batch_size, nn]
    extended_nn_weight = jnp.array([method_kwargs["base_affinity"]])

    # Update banks
    method_state["dataset_feature"] = (
        method_state["dataset_feature"]
        .at[batch_indices]
        .set(model_outputs.embedding)
    )
    method_state["dataset_probability"] = (
        method_state["dataset_probability"]
        .at[batch_indices]
        .set(logit2proba(model_outputs.label))
    )
    return adaptation_state, {
        "nn_weight": flax_utils.replicate(nn_weight),
        "extended_nn_weight": flax_utils.replicate(extended_nn_weight),
        "nn_probability": flax_utils.replicate(nn_probability),
        "extended_nn_probability": flax_utils.replicate(
            extended_nn_probability
        ),
    }

  def get_adaptation_metrics(
      self, supervised: bool, multi_label: bool, **method_kwargs
  ) -> type[clu_metrics.Collection]:
    """Obtain metrics that will be monitored during adaptation.

    Args:
      supervised: Whether the problem is supervised. Only used to know if we can
        track supervised metrics, such as accuracy.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Method's kwargs.

    Returns:
      A collection of metrics.
    """
    metrics_dict = vars(
        adapt.get_common_metrics(supervised=supervised, multi_label=multi_label)
    )["__annotations__"]

    if multi_label:
      metrics_dict["main_loss"] = NRCMultiLoss
    else:
      metrics_dict["main_loss"] = NRCLoss
    return clu_metrics.Collection.create(**metrics_dict)
