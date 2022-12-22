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

"""Our adaptation of the Noisy Student method for SFDA."""

from chirp.projects.sfda import adapt
from chirp.projects.sfda import losses
from chirp.projects.sfda import method_utils
from chirp.projects.sfda import model_utils
from clu import metrics as clu_metrics
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import tensorflow as tf


class DropoutStudent(adapt.SFDAMethod):
  """Our adaptation of the Noisy Student method for SFDA.

  As opposed to the original method, this adpatation only uses a single network.
  The teacher produces predictions from clean images, that the student tries to
  match using Dropout as sole source of noise. In the end, Dropout Student
  is equivalent to NOTELA when setting the weight controlling the Laplacian
  regularization to 0.
  """

  _CITATION = (
      "Xie, Qizhe, et al. 'Self-training with noisy student improves imagenet "
      "classification.' Proceedings of the IEEE/CVF conference on computer "
      "vision and pattern recognition. 2020.")

  def compute_pseudo_label(self,
                           probabilities: jnp.ndarray,
                           multi_label: bool,
                           alpha: float,
                           normalize_pseudo_labels: bool = True) -> jnp.ndarray:
    """Compute the pseudo-labels from the model's probabilities.

    Args:
      probabilities: Model's output probabilities. Shape [*, num_classes]
      multi_label: Whether this is a multi-label problem.
      alpha: Weight controlling the 'softness' of pseudo-labels.
      normalize_pseudo_labels: Whether to normalize pseudo-labels to turn them
        into valid probability distributions. This option should be kept to
        True, and only be used for experimental purposes.

    Returns:
      The pseudo-labels.
    """
    pseudo_labels = jax.lax.stop_gradient(probabilities)
    if multi_label:
      pseudo_labels = jnp.stack([1 - pseudo_labels, pseudo_labels], axis=-1)
    pseudo_labels = pseudo_labels**(1 / alpha)
    if normalize_pseudo_labels:
      pseudo_labels /= pseudo_labels.sum(-1, keepdims=True)
    if multi_label:
      pseudo_labels = pseudo_labels[
          ..., -1]  # we only keep the 'positive' probability
    return pseudo_labels

  def before_epoch(self, key: jax.random.PRNGKeyArray,
                   model_bundle: model_utils.ModelBundle,
                   adaptation_state: adapt.AdaptationState,
                   adaptation_dataset: tf.data.Dataset,
                   modality: adapt.Modality, multi_label: bool,
                   **method_kwargs) -> adapt.AdaptationState:
    """Compute the pseudo-labels when used in 'offline mode'.

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

      # Compute pseudo-labels that will be used during the next epoch of
      # adaptation.
      forward_result = method_utils.forward_dataset(
          dataset=adaptation_dataset,
          adaptation_state=adaptation_state,
          model_bundle=model_bundle,
          modality=modality,
          multi_label=multi_label,
          use_batch_statistics=method_kwargs["update_bn_statistics"])

      sample_ids = forward_result["id"]
      method_state = {
          "pseudo_label":
              self.compute_pseudo_label(
                  forward_result["proba"],
                  multi_label=multi_label,
                  alpha=method_kwargs["alpha"],
                  normalize_pseudo_labels=method_kwargs[
                      "normalize_pseudo_labels"]),
          "id2index": {sample_ids[i]: i for i in range(len(sample_ids))},
      }
      adaptation_state = adaptation_state.replace(method_state=method_state)
    return adaptation_state

  def before_iter(
      self, key: jax.random.PRNGKeyArray, batch: dict[str, np.ndarray],
      adaptation_state: adapt.AdaptationState,
      model_bundle: model_utils.ModelBundle, modality: adapt.Modality,
      multi_label: bool,
      **method_kwargs) -> tuple[adapt.AdaptationState, dict[str, jnp.ndarray]]:
    """Grab or compute the pseudo-labels for the current batch.

    In the offline mode, we only grab pre-computed pseudo-labels from the
    pseudo_label memory. In the online mode, we compute the pseudo-labels
    using the current batch of data.

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
    """

    if method_kwargs["online_pl_updates"]:

      # In the online version, we compute the pseudo-labels on-the-go.
      model_outputs = method_utils.batch_forward(
          adapt.keep_jax_types(batch), adaptation_state.model_state,
          adaptation_state.model_params, model_bundle.model, modality,
          method_kwargs["update_bn_statistics"])
      model_outputs = flax_utils.unreplicate(model_outputs)
      logit2proba = nn.sigmoid if multi_label else nn.softmax
      pseudo_label = self.compute_pseudo_label(
          logit2proba(model_outputs.label), multi_label, method_kwargs["alpha"],
          method_kwargs["normalize_pseudo_labels"])
    else:
      # In the offline version, we simply grab the pseudo-labels that were
      # computed before the epoch.
      method_state = flax_utils.unreplicate(adaptation_state.method_state)
      batch_indexes = np.array([
          method_state["id2index"][x]
          for x in flax_utils.unreplicate(batch["tfds_id"])
      ])
      pseudo_label = method_state["pseudo_label"][batch_indexes]
    return adaptation_state, {
        "pseudo_label": flax_utils.replicate(pseudo_label)
    }

  def get_adaptation_metrics(self, supervised: bool, multi_label: bool,
                             **method_kwargs) -> type[clu_metrics.Collection]:
    """Obtain metrics that will be monitored during adaptation."""
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
