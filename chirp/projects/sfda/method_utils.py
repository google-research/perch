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

"""Some utils functions shared across methods."""

from typing import Dict, Union, Optional

from absl import logging
from chirp.models import taxonomy_model
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
        f"{features_a.shape} and features_b: {features_b.shape}")
  transpose_b = jnp.swapaxes(features_b, -2,
                             -1)  # [*, feature_dim, batch_size_b]
  return jnp.linalg.norm(
      features_a, axis=-1,
      keepdims=True)**2 - 2 * features_a @ transpose_b + jnp.linalg.norm(
          transpose_b, axis=-2,
          keepdims=True)**2  # [batch_size_a, batch_size_b]


def batch_forward(batch: Dict[str, jnp.ndarray],
                  model_state: flax.core.scope.FrozenVariableDict,
                  params: flax.core.scope.VariableDict,
                  model: nn.Module,
                  modality: adapt.Modality,
                  use_batch_statistics: bool,
                  train: bool = False,
                  key: Optional[jax.random.PRNGKeyArray] = None) -> taxonomy_model.ModelOutputs:
  """Collects the model's output on the current batch of data.

  Args:
    batch: The batch of data.
    model_state: The model's state. Expects a replicated model_state.
    params: The model's parameters. Expects replicated params.
    model: The model.
    modality: The modality used.
    use_batch_statistics: Whether to use BatchNorm's running statistics, or the
      batch's statistics.
    train: Whether to use the model in training mode. Default to False, as this
      function is nominally used to compute pseudo-labels (e.g. Teacher step of
      Notela), which usually removes any source of noise (including dropout).
    key: Jax random key to use for the forward pass in case train is set to
      True.

  Returns:
    The model's output.

  Raises:
    ValueError: In case train is set to True, but no random key is specified.
  """
  if train and key is None:
    raise ValueError("Please specifify a random key when using train=True.")
  rngs = {"dropout": key} if key is not None else None
  @jax.pmap
  def forward(batch, model_state, params, rngs):
    if use_batch_statistics:
      outputs, _ = model.apply({
          "params": params,
          **model_state
      },
                               batch[modality.value],
                               train=train,
                               mutable=list(model_state.keys()),
                               use_running_average=False,
                               rngs=rngs)
    else:
      outputs = model.apply({
          "params": params,
          **model_state
      },
                            batch[modality.value],
                            use_running_average=True,
                            train=train,
                            rngs=rngs)
    return outputs

  return forward(batch, model_state, params, rngs)


def forward_dataset(
    dataset: tf.data.Dataset,
    adaptation_state: adapt.AdaptationState,
    model_bundle: model_utils.ModelBundle,
    modality: adapt.Modality,
    multi_label: bool,
    use_batch_statistics: bool = False,
    only_keep_unmasked_classes: bool = False,
    train: bool = False,
    key: Optional[jax.random.PRNGKeyArray] = None
) -> Dict[str, Union[jnp.ndarray, np.ndarray]]:
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
    only_keep_unmasked_classes: In case 'label_mask' is provided as a key in
      batches of data, this option allows to only store the model's probabilties
      for classes that are not masked. This can result in large memory savings,
      e.g. for the bio-acoustic model where <100 classes are present versus the
      ~11k total species.
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

  # Forward the whole dataset. Store embeddings, samples' ids, labels, and
  # model's probabilities.
  for index, batch in tqdm.tqdm(
      enumerate(dataset.as_numpy_iterator()), total=len(dataset)):
    batch = jax.tree_map(np.asarray, batch)
    if key is not None:
      batch_key, key = jax.random.split(key)
      batch_key = jax.random.split(batch_key, num=jax.local_device_count())
    else:
      batch_key = None
    if "label_mask" in batch and only_keep_unmasked_classes and index == 0:
      # We will use the first sample's label_mask as a reference, and ensure
      # all label_masks are the same.
      reference_mask = flax_utils.unreplicate(batch["label_mask"])[0]
    model_outputs = batch_forward(
        adapt.keep_jax_types(batch), model_state, params, model, modality,
        use_batch_statistics, train, batch_key)
    if "label_mask" in batch and only_keep_unmasked_classes:
      # We make sure that the label_mask is the same for all samples in the
      # dataset.
      if not (jnp.tile(reference_mask, (batch["label_mask"].shape[0], 1))
              == batch["label_mask"]).all():
        raise ValueError("All samples should have the same label_mask for the"
                         "'only_keep_unmasked_classes' option to work"
                         "adequately.")
      # We only keep unmasked classes.
      model_outputs = model_outputs.replace(
          label=model_outputs.label[..., reference_mask.astype(bool)])
    all_ouputs.append(flax_utils.unreplicate(model_outputs))
    all_ids += list(batch["tfds_id"].reshape(-1))

  # Concatenate every list to obtain a single array for each field. Store these
  # arrays in the result dictionary.
  logits2proba = nn.sigmoid if multi_label else nn.softmax
  result = {}
  result["embedding"] = jnp.concatenate([x.embedding for x in all_ouputs],
                                        axis=0)  # [dataset_size, n_dimensions]
  result["proba"] = jnp.concatenate([logits2proba(x.label) for x in all_ouputs],
                                    axis=0)  # [dataset_size, num_classes]
  ids = np.array(all_ids)  # [dataset_size,]

  # Make some verifications.
  if np.unique(ids).shape[0] != ids.shape[0]:
    raise ValueError("Ids should uniquely define each sample.")
  result["id"] = ids
  return result
