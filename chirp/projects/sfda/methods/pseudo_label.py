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

"""A pseudo-labelling baseline with confidence thresholding."""

from chirp.projects.sfda import adapt
from chirp.projects.sfda import losses
from chirp.projects.sfda import method_utils
from chirp.projects.sfda import model_utils
from clu import metrics as clu_metrics
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class PseudoLabel(adapt.SFDAMethod):
  """A pseudo-labelling baseline with confidence thresholding.

  The original paper does not use confidence thresholding, but we include it,
  as it is a popular trick used to help stabilize training with pseudo-labels.
  """

  _CITATION = (
      "Lee, Dong-Hyun. 'Pseudo-label: The simple and efficient semi-supervised"
      " learning method for deep neural networks.' Workshop on challenges in "
      "representation learning, ICML. Vol. 3. No. 2. 2013."
  )

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
    """Compute the pseudo-labels for the current batch.

    Low-confidence samples are masked out when computing the loss. We hereby
    compute the mask to only retain high-confidence samples.

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
      A dictionary containing the pseudo-labels and masks to use for the
        iteration.
    """

    # In the online version, we compute the pseudo-labels on-the-go.
    forward_step = self.cache_get_forward_step(
        model_bundle.model, modality, method_kwargs["update_bn_statistics"]
    )
    model_output = forward_step(  # pytype: disable=wrong-arg-types  # jax-ndarray
        adapt.keep_jax_types(batch),
        adaptation_state.model_state,
        adaptation_state.model_params,
        None,
    )
    reference_label_mask = method_utils.get_label_mask(batch)
    probabilities = adapt.logit2proba(
        model_output.label, reference_label_mask, multi_label
    )  # [1, batch_size, num_classes]
    if multi_label:
      # In the multi-label case, given that each class is treated indepently,
      # we perform masking at a "class level", meaning that within one sample,
      # all class probabilities above a certain threshold will contribute to
      # the loss.
      pseudo_label = (
          probabilities > method_kwargs["confidence_threshold"]
      ).astype(
          jnp.float32
      )  # [batch_size, num_classes]
      pseudo_label_mask = pseudo_label
    else:
      if reference_label_mask is None:
        reference_label_mask = jnp.ones_like(probabilities)
      # In the single-label case, we perform masking at a "sample level",
      # meaning that a sample will only contribute to the loss ifs its maximum
      # probability is above some threshold.
      pseudo_label_mask = (probabilities * reference_label_mask).max(
          -1
      ) > method_kwargs[
          "confidence_threshold"
      ]  # [batch_size]
      num_classes = probabilities.shape[-1]
      pseudo_label = nn.one_hot(
          jnp.argmax(probabilities * reference_label_mask, axis=-1),
          num_classes,
          axis=-1,
      )

    return adaptation_state, {
        "pseudo_label": pseudo_label,
        "pseudo_label_mask": pseudo_label_mask,
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
