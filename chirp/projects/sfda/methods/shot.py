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

"""An implementation of Source HypOthesis Transfer (SHOT)."""

from absl import logging
from chirp.projects.sfda import adapt
from chirp.projects.sfda import losses
from chirp.projects.sfda import method_utils
from chirp.projects.sfda import model_utils
from clu import metrics as clu_metrics
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm


@flax.struct.dataclass
class SHOTMultiLabelLoss(clu_metrics.Metric):
  """Computes the loss used in SHOT-full for the multi-label case."""

  probabilities_sum: jnp.array
  entropy_sum: jnp.array
  pl_xent_sum: jnp.array
  label_mask: jnp.array
  n_samples: int
  beta: float

  @classmethod
  def from_model_output(
      cls,
      probabilities: jnp.ndarray,
      pseudo_label: jnp.ndarray,
      beta: float,
      label_mask: jnp.ndarray,
      **_
  ) -> "SHOTMultiLabelLoss":
    """Creates the metric from model's output.

    Args:
      probabilities: The model's binary probabilities. Shape [batch_size,
        num_classes].
      pseudo_label: The pseudo-labels (computed before each epoch). Shape
        [batch_size, num_classes].
      beta: Weight controlling the influence of the pseudo-label cross-entropy.
      label_mask: A mask to control which classes to discard.
      **_:

    Returns:
      An instance of "SHOTMultiLabelLoss".
    """
    entropy_sum = losses.label_binary_ent(
        probabilities=probabilities, label_mask=label_mask
    ).sum(axis=0)
    pl_xent_sum = losses.label_binary_xent(
        probabilities=probabilities, label=pseudo_label, label_mask=label_mask
    ).sum(axis=0)
    probabilities_sum = probabilities.sum(axis=0)

    return cls(
        probabilities_sum=probabilities_sum,
        entropy_sum=entropy_sum,
        pl_xent_sum=pl_xent_sum,
        label_mask=label_mask,
        n_samples=probabilities.shape[0],
        beta=beta,
    )

  def merge(self, other: "SHOTMultiLabelLoss") -> "SHOTMultiLabelLoss":
    return type(self)(
        probabilities_sum=self.probabilities_sum + other.probabilities_sum,
        entropy_sum=self.entropy_sum + other.entropy_sum,
        pl_xent_sum=self.pl_xent_sum + other.pl_xent_sum,
        n_samples=self.n_samples + other.n_samples,
        label_mask=other.label_mask,
        beta=other.beta,
    )

  def compute(self):
    probabilities_marginal = self.probabilities_sum / self.n_samples
    marginal_entropy = losses.label_binary_ent(
        probabilities=probabilities_marginal, label_mask=self.label_mask[0]
    )
    cond_entropy = self.entropy_sum / self.n_samples
    return (
        cond_entropy
        - marginal_entropy
        + self.beta * self.pl_xent_sum / self.n_samples
    )


@flax.struct.dataclass
class SHOTLoss(clu_metrics.Metric):
  """Computes the loss used in SHOT-full for the single-label case."""

  probabilities_sum: jnp.array
  entropy_sum: jnp.array
  pl_xent_sum: jnp.array
  label_mask: jnp.ndarray | None
  n_samples: int
  beta: float

  @classmethod
  def from_model_output(
      cls,
      probabilities: jnp.ndarray,
      pseudo_label: jnp.ndarray,
      label_mask: jnp.array,
      beta: float,
      **_
  ) -> "SHOTLoss":
    entropy_sum = losses.label_ent(
        probabilities=probabilities, label_mask=label_mask
    ).sum(axis=0)
    pl_xent_sum = losses.label_xent(
        probabilities=probabilities, label=pseudo_label, label_mask=label_mask
    ).sum(axis=0)
    probabilities_sum = probabilities.sum(axis=0)

    return cls(
        probabilities_sum=probabilities_sum,
        entropy_sum=entropy_sum,
        pl_xent_sum=pl_xent_sum,
        label_mask=label_mask,
        n_samples=probabilities.shape[0],
        beta=beta,
    )

  def merge(self, other: "SHOTLoss") -> "SHOTLoss":
    return type(self)(
        probabilities_sum=self.probabilities_sum + other.probabilities_sum,
        entropy_sum=self.entropy_sum + other.entropy_sum,
        pl_xent_sum=self.pl_xent_sum + other.pl_xent_sum,
        label_mask=other.label_mask,
        n_samples=self.n_samples + other.n_samples,
        beta=other.beta,
    )

  def compute(self):
    probabilities_marginal = self.probabilities_sum / self.n_samples
    reference_mask = None if self.label_mask is None else self.label_mask[0]
    marginal_entropy = losses.label_ent(probabilities_marginal, reference_mask)
    cond_entropy = self.entropy_sum / self.n_samples
    return (
        cond_entropy
        - marginal_entropy
        + self.beta * self.pl_xent_sum / self.n_samples
    )


class SHOT(adapt.SFDAMethod):
  """SHOT method for SFDA."""

  _CITATION = (
      "Liang, Jian, Dapeng Hu, and Jiashi Feng. 'Do we really need to access "
      "the source data? source hypothesis transfer for unsupervised domain "
      "adaptation.' International Conference on Machine Learning. PMLR, 2020."
  )

  @staticmethod
  def compute_pseudo_label(
      dataset_feature: jnp.ndarray,
      dataset_probability: jnp.ndarray,
      multi_label: bool,
      eps: float = 1e-6,
  ) -> jnp.ndarray:
    """A jax reimplementation of SHOT's pseudo-labelling procedure.

    Original function at https://github.com/tim-learn/SHOT/blob/
    07d0c713e4882e83fded1aff2a447dff77856d64/object/image_target.py#L242.

    Args:
      dataset_feature: The feature for all points in the dataset. Shape
        [dataset_size, feature_dim].
      dataset_probability: Model's probabilities for the current dataset. Shape
        [dataset_size, num_classes].
      multi_label: Whether this is a multi-label problem.
      eps: For numerical stability.

    Returns:
      The pseudo-labels, shape [dataset_size, num_classes]
    """
    classwise_pseudo_label = []
    dataset_probability = dataset_probability.T  # [num_classes, dataset_size]
    if multi_label:
      dataset_probability = jnp.stack(
          [1 - dataset_probability, dataset_probability], axis=1
      )  # [num_classes, probabilities_dim, dataset_size]
    else:
      dataset_probability = jnp.expand_dims(
          dataset_probability, axis=0
      )  # [1, num_classes, dataset_size]
    probabilities_dim = dataset_probability.shape[1]

    # We loop over the classes. Vectorizing this part implies broadcasting
    # `dataset_feature` num_classes times, which may easily lead to OOM.
    for class_probabilities in tqdm.tqdm(
        dataset_probability, total=dataset_probability.shape[0]
    ):
      # Compute initial clusters Eq (4).
      mu_0 = (class_probabilities @ dataset_feature) / (
          class_probabilities.sum(-1, keepdims=True) + eps
      )  # [probabilities_dim, feature_dim]
      # Compute initial pseudo-labels Eq (5)
      dist = method_utils.jax_cdist(
          dataset_feature, mu_0
      )  # [dataset_size, probabilities_dim]

      one_hot_pseudo_label = nn.one_hot(
          dist.argmin(-1), probabilities_dim, axis=-1
      ).transpose()  # [probabilities_dim, dataset_size]

      # Re-Compute clusters and pseudo-labels Eq (6). Equivalent to a second
      # iteration of K-means.
      mu_1 = (one_hot_pseudo_label @ dataset_feature) / (
          one_hot_pseudo_label.sum(-1, keepdims=True) + eps
      )  # [probabilities_dim, feature_dim]
      dist = method_utils.jax_cdist(
          dataset_feature, mu_1
      )  # [dataset_size, probabilities_dim]
      classwise_pseudo_label.append(dist.argmin(-1))  # [dataset_size]
    final_pseudo_label = jnp.stack(
        classwise_pseudo_label, 1
    )  # [dataset_size, num_classes]
    final_pseudo_label = nn.one_hot(
        final_pseudo_label, probabilities_dim, axis=-1
    )  # [dataset_size, num_classes, probabilities_dim]
    if not multi_label:
      assert final_pseudo_label.shape[1] == 1
      final_pseudo_label = jnp.squeeze(
          final_pseudo_label, axis=1
      )  # [dataset_size, probabilities_dim=num_classes]
    else:
      final_pseudo_label = final_pseudo_label[
          ..., -1
      ]  # [dataset_size, num_classes]

    return final_pseudo_label

  def before_epoch(
      self,
      key: jax.random.PRNGKeyArray,
      model_bundle: model_utils.ModelBundle,
      adaptation_state: adapt.AdaptationState,
      adaptation_dataset: tf.data.Dataset,
      modality: adapt.Modality,
      multi_label: bool,
      **method_kwargs
  ) -> adapt.AdaptationState:
    """Compute the pseudo-labels.

    Args:
      key: The jax random key used for random operations in this epoch.
      model_bundle: The ModelBundle used for adaptation.
      adaptation_state: The current state of adaptation.
      adaptation_dataset: The dataset used for adaptation.
      modality: The modality.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Additional method-specific kwargs.

    Returns:
      The adaptation state, with a potentially updated 'method_state' attribute.
    """

    logging.info("Preparing pseudo-labels...")

    # Extract dataset_feature and model's probabilities.
    forward_result = method_utils.forward_dataset(
        dataset=adaptation_dataset,
        adaptation_state=adaptation_state,
        model_bundle=model_bundle,
        modality=modality,
        multi_label=multi_label,
        use_batch_statistics=method_kwargs["update_bn_statistics"],
    )

    # Compute pseudo-labels that will be used during the next epoch of
    # adaptation.
    pseudo_label = SHOT.compute_pseudo_label(
        dataset_feature=forward_result["embedding"],
        dataset_probability=forward_result["proba"],
        multi_label=multi_label,
    )

    sample_ids = forward_result["id"]
    method_state = {
        "pseudo_label": pseudo_label,
        "id2index": {sample_ids[i]: i for i in range(len(sample_ids))},
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
      **method_kwargs
  ) -> tuple[adapt.AdaptationState, dict[str, jnp.ndarray]]:
    """Grab the pseudo-labels from memory for the current batch.

    Args:
      key: The jax random key used for random operations.
      batch: The current batch of data.
      adaptation_state: The current state of adaptation.
      model_bundle: The ModelBundle used for adaptation.
      modality: The current modality.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Additional method-specific kwarg

    Returns:
      The untouched adaptation_state.
      A dictionary containing the pseudo-labels to use for the iteration.
    """
    method_state = flax_utils.unreplicate(adaptation_state.method_state)
    id2index = method_state["id2index"]
    batch_indexes = np.array(
        [id2index[x] for x in flax_utils.unreplicate(batch["tfds_id"])]
    )
    pseudo_label = method_state["pseudo_label"][batch_indexes]

    # pad pseudo-labels to match model output as needed.
    label_mask = method_utils.get_label_mask(batch)
    pseudo_label = method_utils.pad_pseudo_label(
        label_mask, pseudo_label, adaptation_state
    )

    return adaptation_state, {
        "pseudo_label": flax_utils.replicate(pseudo_label)
    }

  def get_adaptation_metrics(
      self, supervised: bool, multi_label: bool, **method_kwargs
  ) -> type[clu_metrics.Collection]:
    """Obtain metrics that will be monitored during adaptation."""
    metrics_dict = vars(
        adapt.get_common_metrics(supervised=supervised, multi_label=multi_label)
    )["__annotations__"]

    if multi_label:
      metrics_dict["main_loss"] = SHOTMultiLabelLoss
    else:
      metrics_dict["main_loss"] = SHOTLoss

    return clu_metrics.Collection.create(**metrics_dict)
