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

"""Test-Time Entropy Minimization (TENT) method."""

import functools

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


class DUST(adapt.SFDAMethod):
  """Adaptation of Dropout-based Uncertainty-driven Self-Training for SFDA.

  Note that DUST itself is not an SFDA method, because it assumes the
  availability of the labelled source data during the adaptation phase. We
  propose a particular way to adapt it to the SFDA setting.
  """

  _CITATION = (
      "Khurana, Sameer, et al. 'Unsupervised domain adaptation forspeech "
      "recognition via uncertainty driven self-training.' ICASSP 2021-2021 "
      "IEEE International Conference on Acoustics, Speech and Signal "
      "Processing(ICASSP). IEEE, 2021."
  )

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
    """Compute the pseudo-labels, the masks and store them in memory.

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
    logging.info("Preparing pseudo-labels...")
    forward_fn = functools.partial(
        method_utils.forward_dataset,
        dataset=adaptation_dataset,
        adaptation_state=adaptation_state,
        model_bundle=model_bundle,
        modality=modality,
        multi_label=multi_label,
        use_batch_statistics=method_kwargs["update_bn_statistics"],
    )

    # Compute model's reference predictions (no dropout used)
    reference_forward_result = forward_fn(train=False)
    reference_probability = reference_forward_result["proba"]

    # We perform multiple noisy forward passes, and keep track of the
    # KL-divergence between the reference predictions and noisy predictions.
    # Note that here we depart from the edit distance proposed in the DUST paper
    # for speech, since the KL-divergence between class label predictions is
    # more appropriate for our use-case.
    kl_distances = []
    kl_fn = losses.label_binary_kl if multi_label else losses.label_kl
    for _ in range(method_kwargs["num_random_passes"]):
      random_pass_key, key = jax.random.split(key)
      noisy_probability = forward_fn(key=random_pass_key, train=True)["proba"]
      kl_distances.append(
          kl_fn(reference_probability, noisy_probability, label_mask=None)
      )

    # We compute the mask by only keeping samples whose maximum kl_divergence
    # observed is lower than a pre-defined threshold.
    pseudo_label_mask = (
        jnp.stack(kl_distances, 0).max(axis=0) < method_kwargs["kl_threshold"]
    )
    sample_ids = reference_forward_result["id"]
    pseudo_label = reference_probability

    # method_state will act as a memory, from which pseudo-labels and masks
    # will be grabbed on-the-go over the next epoch of adaptation.

    method_state = {
        "pseudo_label": pseudo_label,
        "pseudo_label_mask": pseudo_label_mask,
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
    """Grab the pseudo-labels and masks for the current batch.

    Args:
      key: The jax random key used for random operations.
      batch: The current batch of data.
      adaptation_state: The current state of adaptation.
      model_bundle: The ModelBundle used for adaptation.
      modality: The current modality.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Additional method-specific kwarg

    Returns:
      A dictionary containing the pseudo-labels and mask to use for the
      iteration.
    """
    method_state = flax_utils.unreplicate(adaptation_state.method_state)
    id2index = method_state["id2index"]
    batch_indices = np.array(
        [id2index[x] for x in flax_utils.unreplicate(batch["tfds_id"])]
    )

    pseudo_label = method_state["pseudo_label"][batch_indices]
    pseudo_label_mask = method_state["pseudo_label_mask"][batch_indices]

    # pad pseudo-labels to match model output as needed.
    label_mask = method_utils.get_label_mask(batch)
    pseudo_label = method_utils.pad_pseudo_label(
        label_mask, pseudo_label, adaptation_state
    )
    if multi_label:
      pseudo_label_mask = method_utils.pad_pseudo_label(
          label_mask, pseudo_label_mask, adaptation_state
      )

    return adaptation_state, {
        "pseudo_label": flax_utils.replicate(pseudo_label),
        "pseudo_label_mask": flax_utils.replicate(pseudo_label_mask),
    }

  def get_adaptation_metrics(
      self, supervised: bool, multi_label: bool, **method_kwargs
  ) -> type[clu_metrics.Collection]:
    """Obtain metrics that will be monitored during adaptation."""
    metrics_dict = vars(
        adapt.get_common_metrics(supervised=supervised, multi_label=multi_label)
    )["__annotations__"]

    def single_label_loss_fn(
        probabilities, pseudo_label, pseudo_label_mask, label_mask, **_
    ):
      pl_xent = losses.label_xent(
          probabilities=probabilities,
          label=pseudo_label,
          label_mask=label_mask,
          sample_mask=pseudo_label_mask,
      )
      return pl_xent

    def multi_label_loss_fn(
        probabilities: jnp.ndarray,
        pseudo_label: jnp.ndarray,
        pseudo_label_mask: jnp.ndarray,
        label_mask: jnp.ndarray,
        **_
    ):
      # Sample's probabilities that end up contributing to the final computation
      # are those left unmasked by label_mask (defined by the target domain and
      # restricting the set of possible species) and are confident enough (
      # left unmasked by pseudo_label_mask).
      pl_xent = losses.label_binary_xent(
          probabilities=probabilities,
          label=pseudo_label,
          label_mask=label_mask * pseudo_label_mask,
      )
      return pl_xent

    loss_fn = multi_label_loss_fn if multi_label else single_label_loss_fn
    metrics_dict["main_loss"] = clu_metrics.Average.from_fun(loss_fn)
    return clu_metrics.Collection.create(**metrics_dict)
