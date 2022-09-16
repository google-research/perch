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

"""HuBERT model."""
from typing import Any, Dict, Tuple, Optional, Union

import flax
from flax import linen as nn
import jax
from jax import numpy as jnp


@flax.struct.dataclass
class ModelOutputs:
  embedding: jnp.ndarray
  logits: jnp.ndarray
  targets: jnp.ndarray
  mask_idc: jnp.ndarray
  quantization_loss: jnp.ndarray
  label: jnp.ndarray
  genus: Optional[jnp.ndarray] = None
  family: Optional[jnp.ndarray] = None
  order: Optional[jnp.ndarray] = None


def compute_mask_indices(key: jnp.ndarray,
                         shape: Tuple[int, int],
                         mask_prob: float,
                         mask_length: int,
                         min_masks: int = 0) -> jnp.ndarray:
  """Computes random mask spans for a given shape.

  Args:
    key: The key for random operations.
    shape: The shape of the mask that will be computed. A tuple of two elements,
      corresponding to the batch size and the number of frames.
    mask_prob: The probability for each token to be chosen as the starting index
      of a 'masked span'.
    mask_length: The length of each 'masked span'.
    min_masks: The minimum number of masked spans.

  Returns:
    mask: A boolean jnp.array that has the same shape as `shape`.
  """
  bsz, sz = shape
  key, subkey = jax.random.split(key)

  # `num_mask` is the number of 'masked spans' for each sample in the batch.
  # A random number is added for probabilistic rounding. We use the 'static'
  # strategy where each sample in the batch has the same number of masked spans.
  rounding_offset = jax.random.uniform(subkey, shape=(bsz,))
  key, subkey = jax.random.split(key)
  num_mask = mask_prob * sz / jnp.array(mask_length, float) + rounding_offset
  num_mask = jnp.full(bsz, num_mask).astype(int)
  max_masks = sz - mask_length + 1
  num_mask = jnp.clip(num_mask, a_min=min_masks, a_max=max_masks)

  # First, sample a set of start indices for the max possible number of masks.
  # Do this sampling separately for each batch sample, to allow `replace`=False.
  max_start_index = sz - mask_length
  mask_idc = []
  for _ in range(bsz):
    mask_idc.append(
        jax.random.choice(
            subkey, max_start_index + 1, shape=(max_masks,), replace=False))
    key, subkey = jax.random.split(key)
  mask_idc = jnp.stack(mask_idc, axis=0)

  # Now filter these starting indices to `num_mask` 'active' ones. This is done
  # by replacing the 'inactive' ones to start at some index that is beyond the
  # length of the sequence. The scatter operation later will disregard these.
  mask_idc = jnp.reshape(mask_idc, [-1])
  inactive_start_idx = sz
  a = jnp.array([a % max_masks for a in jnp.arange(max_masks * bsz)])
  num_mask = jnp.reshape(
      jnp.repeat(jnp.expand_dims(num_mask, 1), axis=1, repeats=max_masks), [-1])
  mask_idc = jnp.where(a < num_mask, mask_idc, inactive_start_idx)

  # Add the offsets, to get all masked indices of each span.
  mask_idc = jnp.concatenate(
      [mask_idc + offset for offset in range(mask_length)])

  # Prepare the `scatter_indices`, i.e. the positions of a (bsz, sz) array that
  # will be set to 1s in the binary mask that will be returned.
  batch_inds = jnp.reshape(
      jnp.repeat(
          jnp.expand_dims(jnp.arange(bsz), 1), axis=1, repeats=max_masks), [-1])
  batch_inds = jnp.reshape(
      jnp.repeat(jnp.expand_dims(batch_inds, 0), axis=0, repeats=mask_length),
      [-1])
  scatter_indices = jnp.stack((batch_inds, mask_idc), axis=1)

  mask = jax.lax.scatter(
      jnp.zeros((bsz, sz)).astype(int), scatter_indices,
      jnp.ones_like((mask_idc)).astype(int),
      jax.lax.ScatterDimensionNumbers(
          update_window_dims=(),
          inserted_window_dims=(0, 1),
          scatter_dims_to_operand_dims=(0, 1)))
  return mask


class HuBERTModel(nn.Module):
  """HuBERT model.

  Attributes:
    num_classes: Number of classes for each output head. These are used to train
      supervised readout layer for evaluation only. The representation is
      learned in a purely self-supervised manner.
    early_feature_extractor: A network (e.g., a 2D convolutional network) that
      takes spectrograms and returns feature vectors. Quantization is performed
      on the features produced by this feature extractor.
    late_feature_extractor: A network (e.g., a stack of Conformer blocks) that
      takes "early" features and transforms them into higher-level features.
    quantizer: A network that takes spectrograms and returns a codebook, the
      assignments of inputs to codes, and a loss for training it.
    frontend: The frontend to use to generate features.
    mask_config: The config for generating masks.
    final_dim: The dimensionality after the final projection layer.
    logit_temp: The temperature to use for the logits of which cluster each
      timestep belongs to.
    alpha: The weight of the masked loss in the combination of the masked and
      unmasked losses for HuBERT. By default it's 1, considering only masked.
    taxonomy_loss_weight: Weight for taxonomic label losses. These are used to
      train supervised readout layer for evaluation only. The representation is
      learned in a purely self-supervised manner.
    quant_loss_mult: Multiplier for the quantizer loss in the overall loss used
      for training.
  """
  num_classes: Dict[str, int]
  early_feature_extractor: nn.Module
  late_feature_extractor: nn.Module
  quantizer: nn.Module
  frontend: nn.Module
  mask_config: Dict[str, Any]
  taxonomy_loss_weight: float
  final_dim: int = 512
  logit_temp: float = 0.1
  alpha: float = 1.0
  quant_loss_mult: float = 1.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool,
               mask_key: Union[jnp.ndarray, None]) -> ModelOutputs:
    """Apply the HuBERT model.

    bsz: batch size.
    sz: number of frames (timesteps).
    csz: number of channels.
    nc: number of centroids.
    ns: number of sections of the product quantizer (if applicable).

    Args:
      inputs: Audio of shape `(bsz, time)`.
      train: Whether we're in training mode (affects batch norm, dropout and
        whether masking is applied).
      mask_key: A jnp.array that serves as the key for sampling masks. It can be
        None if `train` is False since no mask is applied in that case.

    Returns:
      Logits for which cluster each timestep belongs to (per section of the
        product quantizer, if applicable).
    """
    if train and mask_key is None:
      raise ValueError("During training mode, `mask_key` should not be None.")

    model_outputs = {}

    # Pass x through the frontend and the "early" feature extractor.
    x = self.frontend(inputs)
    x = self.early_feature_extractor(x, train=train)

    bsz, sz, csz = x.shape
    nc = self.quantizer.num_centroids

    # The learnable mask token.
    mask_emb = self.param("mask_emb", nn.initializers.uniform(), (csz,))

    # Get the codes, quantization targets and quantizer loss.
    # codes: the cluster embeddings of shape [nc, csz].
    # nn_idx: the (dense) labels of shape [bsz, sz].
    # targets will be one-hot labels of shape [bsz, sz, nc].
    _, quantization_loss, nn_idx, codes = self.quantizer(x)
    targets = jax.nn.one_hot(nn_idx, nc)
    model_outputs["targets"] = targets
    model_outputs["quantization_loss"] = quantization_loss

    # Project the centroids. [nc, final_dim].
    codes_pj = nn.Dense(self.final_dim, name="project_codes")(codes)

    # Stop gradients so that HuBERT doesn't train the early feature extractor.
    x = jax.lax.stop_gradient(x)

    # Get the corrupted x, where the features are replaced with the learnable
    # masked embedding for the positions that are chosen to be masked, if we are
    # in training mode.
    mask_idc = jnp.zeros((bsz, sz))
    if train:
      mask_idc = compute_mask_indices(
          mask_key, shape=(bsz, sz), **self.mask_config)
    model_outputs["mask_idc"] = mask_idc
    mask_idc_exp = jnp.repeat(jnp.expand_dims(mask_idc, 2), repeats=csz, axis=2)
    x = jnp.where(mask_idc_exp > 0, mask_emb, x)

    # Pass the corrupted x through the "late" feature extractor.
    x = self.late_feature_extractor(x, train=train)
    _, _, csz = x.shape
    model_outputs["embedding"] = x

    # A linear head for supervised classification on top of HuBERT embeddings.
    # The gradients of this loss will not propagate to train the representation
    # (the representation is trained purely self-supervised). This is useful for
    # evaluation of the representations (via classification on validation set).
    for k, n in self.num_classes.items():
      # To be consistent with the implementation of conformers in the supervised
      # `TaxonomyModel`, we average over the time dimension before the readout
      # layer, to collapse x from [bsz, sz, csz] to [bsz, csz]. But in this case
      # we only average over the *unmasked* frames.

      # x_filtered_zeros has 0s in place of masked embeddings, while keeping
      # only the unmasked embeddings intact. [bsz, sz, csz].
      mask_idc_exp = jnp.repeat(
          jnp.expand_dims(mask_idc, 2), repeats=csz, axis=2)
      x_filtered = jnp.where(mask_idc_exp, 0, x)
      unmasked_mean = jnp.sum(
          x_filtered, axis=1) / jnp.sum(
              mask_idc_exp == 0, axis=1)

      # Stop grad to ensure supervision doesn't leak into representations.
      model_outputs[k] = nn.Dense(n)(jax.lax.stop_gradient(unmasked_mean))

    # Final projection layer that projects embeddings to `final_dim`. These
    # projected inputs will be used for the nearest-neighbour search with codes.
    x = nn.Dense(self.final_dim, name="project_embeddings")(x)

    # Predict the code of each timestep using cosine similarity between the
    # projected embeddings and the projected codes.
    # x is [bsz, sz, final_dim].
    # codes_pj is [nc, final_dim].
    # logits will be [bsz, sz, nc].
    codes_pj /= (jnp.linalg.norm(codes_pj, axis=-1, keepdims=True) + 1e-5)
    x /= (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5)
    logits = jnp.dot(x, codes_pj.T)
    # TODO(etriantafillou): experiment with learnable temperature.
    logits /= self.logit_temp
    model_outputs["logits"] = logits

    return ModelOutputs(**model_outputs)
