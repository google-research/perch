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

"""Some utils functions shared across methods."""

from typing import Callable

from absl import logging
from chirp.models import output
from chirp.projects.sfda import adapt
from chirp.projects.sfda import model_utils
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm


ForwardStepType = Callable[
    [
        dict[str, jnp.ndarray],
        flax.core.scope.FrozenVariableDict,
        flax.core.scope.VariableDict,
        jax.random.PRNGKeyArray | None,
    ],
    output.ClassifierOutput,
]


@jax.jit
def jax_cdist(features_a: jnp.array, features_b: jnp.array) -> jnp.array:
  """A jax equivalent of scipy.spatial.distance.cdist.

  Computes the pairwise squared euclidean distance between each pair of features
  from features_a and features_b.

  Args:
    features_a: The first batch of features, expected shape [*, batch_size_a,
      feature_dim]
    features_b: The second batch of features, expected shape [*, batch_size_b,
      feature_dim]

  Returns:
    The pairwise squared euclidean distance between each pair of features from
    features_a and features_b. Shape [*, batch_size_a, batch_size_b]

  Raises:
    ValueError: If the shape of features_a's last dimension does not match the
      shape of feature_b's last dimension.
  """
  if features_a.shape[-1] != features_b.shape[-1]:
    raise ValueError(
        "The feature dimension should be the same. Currently features_a: "
        f"{features_a.shape} and features_b: {features_b.shape}"
    )
  feature_dim = features_a.shape[-1]

  flat_features_a = jnp.reshape(features_a, [-1, feature_dim])
  flat_features_b = jnp.reshape(features_b, [-1, feature_dim])
  flat_transpose_b = flat_features_b.T
  distances = (
      jnp.sum(jnp.square(flat_features_a), 1, keepdims=True)
      - 2 * jnp.matmul(flat_features_a, flat_transpose_b)
      + jnp.sum(jnp.square(flat_transpose_b), 0, keepdims=True)
  )
  return distances


def forward_dataset(
    dataset: tf.data.Dataset,
    adaptation_state: adapt.AdaptationState,
    model_bundle: model_utils.ModelBundle,
    modality: adapt.Modality,
    multi_label: bool,
    use_batch_statistics: bool = False,
    train: bool = False,
    key: jax.random.PRNGKeyArray | None = None,
) -> dict[str, jnp.ndarray | np.ndarray]:
  """Fowards a dataset through a given model.

  Args:
    dataset: The dataset to extract from.
    adaptation_state: The current adaptation state, including the model's state
      and parameters used for extraction.
    model_bundle: The current ModelBundle.
    modality: The current data modality.
    multi_label: Whether this is a multi-label problem. This affects how model's
      probabilities are packaged.
    use_batch_statistics: Whether to use batch's statistics for BatchNorm layers
      during feature extraction.
    train: Whether to use dropout or not during the forward pass.
    key: The random key to use if train is set to True.

  Returns:
    A dictionnary with the following keys:
      -embeddings: The extracted embeddings of shape [dataset_size,
       embedding_dimension].
      -proba: The output classwise probabities of shape [dataset_size,
       num_classes].
      -ids: The ids of the examples extracted, consistent with  embeddings and
       proba, of shape [N]. ids[i] corresponds to embeddings[i] and proba[i].

  Raises:
    ValueError: In case the ids do not uniquely identify each sample, or if
      the samples don't have the same label_mask.
  """
  logging.info("Starting feature extraction...")
  all_ouputs = []
  all_ids = []  # used to identify all samples
  model_state = flax_utils.replicate(adaptation_state.model_state)
  params = flax_utils.replicate(adaptation_state.model_params)
  model = model_bundle.model
  only_keep_unmasked_classes = adaptation_state.restrict_classes
  forward_step = adapt.batch_forward(
      model, modality, use_batch_statistics, train
  )

  # Forward the whole dataset. Store embeddings, samples' ids, labels, and
  # model's probabilities.
  for index, batch in tqdm.tqdm(
      enumerate(dataset.as_numpy_iterator()), total=len(dataset)
  ):
    batch = jax.tree_map(np.asarray, batch)
    if key is not None:
      batch_key, key = jax.random.split(key)
      batch_key = jax.random.split(batch_key, num=jax.local_device_count())
      batch_key = {"dropout": batch_key}
    else:
      batch_key = None
    if "label_mask" in batch and only_keep_unmasked_classes and index == 0:
      # We will use the first sample's label_mask as a reference, and ensure
      # all label_masks are the same.
      reference_mask = flax_utils.unreplicate(batch["label_mask"])[0]
    model_outputs = forward_step(  # pytype: disable=wrong-arg-types  # jax-ndarray
        adapt.keep_jax_types(batch), model_state, params, batch_key
    )
    if "label_mask" in batch and only_keep_unmasked_classes:
      # We make sure that the label_mask is the same for all samples in the
      # dataset.
      if not (
          jnp.tile(reference_mask, (batch["label_mask"].shape[0], 1))
          == batch["label_mask"]
      ).all():
        raise ValueError(
            "All samples should have the same label_mask for the"
            "'only_keep_unmasked_classes' option to work"
            "adequately."
        )
      # We only keep unmasked classes.
      model_outputs = model_outputs.replace(
          label=model_outputs.label[..., reference_mask.astype(bool)]
      )
    all_ouputs.append(flax_utils.unreplicate(model_outputs))
    all_ids += list(batch["tfds_id"].reshape(-1))

  # Concatenate every list to obtain a single array for each field. Store these
  # arrays in the result dictionary.
  logits2proba = nn.sigmoid if multi_label else nn.softmax
  result = {}
  result["embedding"] = jnp.concatenate(
      [x.embedding for x in all_ouputs], axis=0
  )  # [dataset_size, n_dimensions]
  result["proba"] = jnp.concatenate(
      [logits2proba(x.label) for x in all_ouputs], axis=0
  )  # [dataset_size, num_classes]
  ids = np.array(all_ids)  # [dataset_size,]

  # Make some verifications.
  if np.unique(ids).shape[0] != ids.shape[0]:
    raise ValueError("Ids should uniquely define each sample.")
  result["id"] = ids
  if "label_mask" in batch:
    result["label_mask"] = reference_mask
  return result


def maybe_restrict_labels(
    model_outputs, reference_label_mask, adaptation_state
):
  """Restrict model_outputs to target classes, if appropriate."""
  if not adaptation_state.restrict_classes:
    return model_outputs
  if reference_label_mask is None:
    raise ValueError("Asked to restrict classes, but no label mask provided.")
  # We restrict the model's logits to the classes that appear in the
  # current dataset to ensure compatibility with
  # method_state["dataset_proba"].
  model_outputs = model_outputs.replace(
      label=model_outputs.label[..., reference_label_mask.astype(bool)]
  )
  return model_outputs


def get_label_mask(batch) -> jnp.ndarray | None:
  if "label_mask" in batch:
    label_mask = flax_utils.unreplicate(batch["label_mask"])
    reference_label_mask = label_mask[0]  # [num_classes]
    # Ensure that the label_mask is the same for all samples.
    assert (
        jnp.tile(reference_label_mask, (label_mask.shape[0], 1)) == label_mask
    ).all()
  else:
    reference_label_mask = None
  return reference_label_mask


def pad_pseudo_label(
    reference_label_mask: jnp.ndarray | None,
    pseudo_label: jnp.ndarray,
    adaptation_state: adapt.AdaptationState,
) -> jnp.ndarray:
  """Pads pseudo-labels back to the global probability space.

  Args:
    reference_label_mask: The mask indicating which 'global' classes are used
      for the adaptation, shape [num_classes].
    pseudo_label: Pseudo-label, expressed in a potentially reduced probability
      space, shape [batch_size, label_mask.sum()].
    adaptation_state: The adaptation state.

  Returns:
    The zero-padded pseudo-labels, of shape [batch_size, num_classes]

  Raises:
    ValueError: If pseudo_label's last dimension does not match the number of
      classes used for adaptation, as indicated by label_mask
  """
  if not adaptation_state.restrict_classes:
    return pseudo_label
  if reference_label_mask is None:
    raise ValueError("Asked to pad pseudolabels, but no label mask provided.")
  if reference_label_mask.ndim != 1:
    raise ValueError(
        "Expecting a vector for label_mask. Current shape is"
        f" {reference_label_mask.shape}"
    )
  batch_size = pseudo_label.shape[0]
  num_classes_used = reference_label_mask.sum()
  num_classes_total = reference_label_mask.shape[0]
  if pseudo_label.shape[-1] != num_classes_used:
    raise ValueError(
        "Pseudo-labels should be expressed in the same"
        "restricted set of classes provided by the label_mask."
        "Currently, label_mask indicates that "
        f"{num_classes_used} should be used, but pseudo_label "
        f"is defined over {pseudo_label.shape[-1]} classes."
    )
  padded_pseudo_label = jnp.zeros((batch_size, num_classes_total))
  col_index = jnp.tile(jnp.where(reference_label_mask)[0], batch_size)
  row_index = jnp.repeat(jnp.arange(batch_size), num_classes_used)
  return padded_pseudo_label.at[(row_index, col_index)].set(
      pseudo_label.flatten()
  )
