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

"""HuBERT model."""
import enum
from typing import Any, Sequence

from chirp.models import conformer
from chirp.models import layers
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp


@flax.struct.dataclass
class HubertOutput:
  embedding: list[jnp.ndarray]
  logits: list[jnp.ndarray]
  targets: list[jnp.ndarray]
  mask_idc: jnp.ndarray
  quantization_loss: list[jnp.ndarray]
  label: list[jnp.ndarray]
  genus: list[jnp.ndarray] | None = None
  family: list[jnp.ndarray] | None = None
  order: list[jnp.ndarray] | None = None


class QuantizerPoints(enum.Enum):
  """A point in the architecture to add a quantizer."""

  FRONTEND = -2
  EARLY_FS = -1


def compute_mask_indices(
    key: jnp.ndarray,
    shape: tuple[int, int],
    mask_prob: float,
    mask_length: int,
    min_masks: int = 0,
) -> jnp.ndarray:
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
            subkey, max_start_index + 1, shape=(max_masks,), replace=False
        )
    )
    key, subkey = jax.random.split(key)
  mask_idc = jnp.stack(mask_idc, axis=0)

  # Now filter these starting indices to `num_mask` 'active' ones. This is done
  # by replacing the 'inactive' ones to start at some index that is beyond the
  # length of the sequence. The scatter operation later will disregard these.
  mask_idc = jnp.reshape(mask_idc, [-1])
  inactive_start_idx = sz
  a = jnp.array([a % max_masks for a in jnp.arange(max_masks * bsz)])
  num_mask = jnp.reshape(
      jnp.repeat(jnp.expand_dims(num_mask, 1), axis=1, repeats=max_masks), [-1]
  )
  mask_idc = jnp.where(a < num_mask, mask_idc, inactive_start_idx)

  # Add the offsets, to get all masked indices of each span.
  mask_idc = jnp.concatenate(
      [mask_idc + offset for offset in range(mask_length)]
  )

  # Prepare the `scatter_indices`, i.e. the positions of a (bsz, sz) array that
  # will be set to 1s in the binary mask that will be returned.
  batch_inds = jnp.reshape(
      jnp.repeat(
          jnp.expand_dims(jnp.arange(bsz), 1), axis=1, repeats=max_masks
      ),
      [-1],
  )
  batch_inds = jnp.reshape(
      jnp.repeat(jnp.expand_dims(batch_inds, 0), axis=0, repeats=mask_length),
      [-1],
  )
  scatter_indices = jnp.stack((batch_inds, mask_idc), axis=1)

  mask = jax.lax.scatter(
      jnp.zeros((bsz, sz)).astype(int),
      scatter_indices,
      jnp.ones_like((mask_idc)).astype(int),
      jax.lax.ScatterDimensionNumbers(
          update_window_dims=(),
          inserted_window_dims=(0, 1),
          scatter_dims_to_operand_dims=(0, 1),
      ),
  )
  return mask


@flax.struct.dataclass
class QuantizerBundle:
  quantization_loss: jnp.ndarray
  targets: jnp.ndarray
  codebook: jnp.ndarray
  projected_feature_codes: jnp.ndarray


class HuBERTEval(nn.Module):
  """HuBERT model for evaluation.

  Attributes:
    early_feature_extractor: A network (e.g., a 2D convolutional network) that
      takes spectrograms and returns feature vectors. Quantization is performed
      on the features produced by this feature extractor.
    late_feature_extractor: A network (e.g., a stack of Conformer blocks) that
      takes "early" features and returns a sequence of Jax ndarrays that contain
      increasingly higher-level features.
    frontend: The frontend to use to generate features.
    use_raw_audio: Whether to feed raw audio into the feature extractor to get
      HuBERT's predictions (as opposed to audio processed with a frontend). The
      current best configuration sets this option to True but performs
      quantization after the frontend to obtain targets for HuBERT.
    add_positional_embeddings: Whether to add positional embeddings to the late
      feature extractor.
  """

  early_feature_extractor: nn.Module | None
  late_feature_extractor: nn.Module
  frontend: nn.Module | None = None
  use_raw_audio: bool = True
  add_positional_embeddings: bool = False

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, block_to_readout=6):
    """Forward pass through the HuBERT model for evaluation.

    bsz: batch size.
    sz: number of frames (timesteps).
    csz: number of channels.

    Args:
      inputs: Audio of shape `(bsz, sz)`.
      block_to_readout: The integer representing the block of the late feature
        extractor from which to return embeddings.

    Returns:
      The late feature extractor embeddings of after block `block_to_readout`.
    """
    if self.use_raw_audio:
      # Raw audio (no frontend) is fed through the early feature extractor.
      early_fs_inputs = jnp.expand_dims(inputs, -1)  # (bsz, sz, 1)
      x_earlyfs = self.early_feature_extractor(early_fs_inputs, train=False)
      x = x_earlyfs
    else:
      # Process audio with a frontend before the early feature extractor.
      x_frontend = self.frontend(inputs)  # (bsz, sz, csz)
      if self.early_feature_extractor is not None:
        x_earlyfs = self.early_feature_extractor(x_frontend, train=False)
      x = x_earlyfs

    _, sz, csz = x.shape
    if self.add_positional_embeddings:
      x = x + conformer.PositionalEmbedding(embedding_dims=csz)(seq_length=sz)

    # Pass x through the "late" feature extractor. Returns a list of x's for the
    # different "readout points".
    x_list = self.late_feature_extractor(
        x, train=False, return_intermediate_list=True
    )
    if block_to_readout < 0 or block_to_readout >= len(x_list):
      raise ValueError(
          "The `block_to_readout` should be in the range "
          "[0, len(x_list)) where x_list is the list that the "
          "late feature extractor returns. But got {} and "
          "len(x_list) is {}".format(block_to_readout, len(x_list))
      )

    # Stop gradient so that a classifier is trained on frozen features for
    # evaluation purposes.
    return jax.lax.stop_gradient(x_list[block_to_readout])


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
      takes "early" features and returns a sequence of Jax ndarrays that contain
      increasingly higher-level features.
    quantizer: A list of quantizer networks, each of which returns a codebook,
      assignments of inputs to codes, and a loss for training it. This list may
      contain only a single element, or several in the case of quantizing in
      different feature spaces.
    frontend: The frontend to use to generate features.
    use_raw_audio: Whether to feed raw audio into the feature extractor to get
      HuBERT's predictions (as opposed to audio processed with a frontend). The
      current best configuration sets this option to True but performs
      quantization after the frontend to obtain targets for HuBERT.
    mask_config: The config for generating masks.
    classifier_config: The config for the classifier.
    taxonomy_loss_weight: Weight for taxonomic label losses. These are used to
      train supervised readout layer for evaluation only. The representation is
      learned in a purely self-supervised manner.
    readout_points: A List of indices of late feature extractor blocks after
      which to add a readout layer (for classification). The allowed values are
      in the range [0, len(x_list)) where x_list is the list of Jax ndarrays
      returned by the late feature extractor.
    quantizer_points: A list of integers indicating where to quantize. The
      allowed values are ints in the range [0, len(x_list)) where x_list is the
      list of ndarrays returned by the late feature extractor and values of the
      QuantizerPoints class. Specifically, any non-negative integer represents
      quantizing after the late feature extractor block with that integer index,
      whereas QuantizerPoints values also allow to quantize right after the
      frontend (QuantizerPoints.FRONTEND) and right after the early feature
      extractor (QuantizerPoints.EARLY_FS).
    final_dim: The dimensionality after the final projection layer.
    logit_temp: The temperature to use for the logits of which cluster each
      timestep belongs to.
    alpha: The weight of the masked loss in the combination of the masked and
      unmasked losses for HuBERT. By default it's 1, considering only masked.
    stop_gradient_earlyfs: Whether to stop gradient after the early feature
      extractor.
    omit_classifier_stop_grads: Optionally, a list of integers indicating which
      of the readout points to omit the stop-gradient for. Specifically, a
      classifier is added to each readout point, and typically a stop-gradient
      is used to prevent the classifier from modifying the representations (this
      happens by default, if this argument isn't provided, or if it's None). If
      provided, specifies the index (or indices) of readout locations where that
      stop-gradient operation will be omitted. This allows, for instance,
      supervised finetuning of HuBERT representations, and semi-supervised
      learning.
    add_positional_embeddings: Whether to add positional embeddings to the late
      feature extractor.
  """

  num_classes: dict[str, int]
  early_feature_extractor: nn.Module | None
  late_feature_extractor: nn.Module
  quantizer: list[nn.Module]
  frontend: nn.Module
  use_raw_audio: bool
  mask_config: dict[str, Any]
  classifier_config: dict[str, Any]
  taxonomy_loss_weight: float
  readout_points: list[int]
  quantizer_points: Sequence[int]
  final_dim: int = 512
  logit_temp: float = 0.1
  alpha: float = 1.0
  stop_gradient_earlyfs: bool = True
  omit_classifier_stop_grads: Sequence[int] | None = None
  add_positional_embeddings: bool = False

  def classify(
      self,
      x_list,
      mask_idc,
      per_frame_predictions,
      classify_pool_width,
      classify_stride,
      classify_features,
      reduction_type,
      classify_from_all,
  ):
    # The gradients of this loss will not propagate to train the representation
    # (the representation is trained purely self-supervised), unless it is
    # requested to omit placing a stop-gradient on the classifier.
    for i in range(len(x_list)):
      if (
          self.omit_classifier_stop_grads is None
          or i not in self.omit_classifier_stop_grads
      ):
        x_list[i] = jax.lax.stop_gradient(x_list[i])
    outputs = {}
    midpt = x_list[-1].shape[-2] // 2  # The middle frame.
    for k, n in self.num_classes.items():
      outputs[k] = []

      # We use separate readout heads on different "levels" of representation.
      for i, x_interm in enumerate(x_list):
        if i not in self.readout_points:
          continue
        csz_ = x_interm.shape[-1]

        if per_frame_predictions:
          # Borrow the classifier from `separation_model.py`.
          x_interm = nn.normalization.LayerNorm(reduction_axes=(-2, -1))(
              x_interm
          )
          x_interm = layers.StridedAutopool(
              0.5,
              classify_pool_width,
              classify_stride,
              padding="SAME",
              name="readout_autopool_{}_{}".format(k, i),
          )(x_interm)
          x_interm = nn.Conv(
              features=classify_features,
              kernel_size=(1,),
              strides=(1,),
              padding="SAME",
              name="readout_conv1_{}_{}".format(k, i),
          )(x_interm)
          x_interm = nn.swish(x_interm)
          per_frame_preds = nn.Conv(
              n, (1,), (1,), "SAME", name="readout_conv2_{}_{}".format(k, i)
          )(x_interm)
          # Now reduce over the time axis to get 1 prediction per *sample*.
          if reduction_type == "AVG":
            reduce_fn = lambda x: jnp.mean(x, axis=-2)
          elif reduction_type == "MAX":
            reduce_fn = lambda x: jnp.max(x, axis=-2)
          elif reduction_type == "MIDPOINT":
            reduce_fn = lambda x: x[..., midpt, :]
          else:
            raise ValueError(f"Reduction {reduction_type} not recognized.")
          outputs[k].append(reduce_fn(per_frame_preds))
        else:
          # Akin to the implementation of conformers in the supervised model,
          # we average over the time dimension before the readout layer, to
          # collapse x from [bsz, sz, csz] to [bsz, csz]. But in this case
          # we only average the *unmasked* frames if `classify_from_all` is off.
          if classify_from_all:
            mean = jnp.mean(x_interm, axis=1)
          else:
            # x_filtered_zeros has 0s in place of masked embeddings, while
            # keeping only the unmasked embeddings intact. [bsz, sz, csz_].
            mask_idc_exp = jnp.repeat(
                jnp.expand_dims(mask_idc, 2), repeats=csz_, axis=2
            )
            x_filtered = jnp.where(mask_idc_exp, 0, x_interm)
            mean = jnp.sum(x_filtered, axis=1) / jnp.sum(
                mask_idc_exp == 0, axis=1
            )

          outputs[k].append(
              nn.Dense(n, name="readout_{}_{}".format(k, i))(mean)
          )
    return outputs

  def add_projected_quantizer(self, x, quantizers, train):
    """Adds a quantizer on top of features x."""
    # Get the next quantizer module.
    quant_index = len(quantizers)
    quantizer = self.quantizer[quant_index]
    nc = quantizer.get_num_centroids()
    ns = quantizer.get_num_sections()

    # Get the codes, quantization targets and quantizer loss.
    quant_outputs = quantizer(x, train)
    # codes: [ns, nc, csz / ns], where ns = 1 if not using PQ.
    codes = quant_outputs.codebook
    # quant_outputs.nn_idx: [ns, bsz, sz].
    # targets: [ns, bsz, sz, nc].
    nn_idx = quant_outputs.nn_idx
    targets = jax.nn.one_hot(nn_idx, nc)

    # Project the centroids.
    # A list of ns many elements that have shape [nc, final_dim].
    codes_pj = [
        nn.Dense(
            self.final_dim, name="codes_proj_{}_{}".format(quant_index, i)
        )(codes[i])
        for i in range(ns)
    ]
    # [ns, nc, final_dim].
    codes_pj = jnp.stack(codes_pj, axis=0)
    quantizers.append(
        QuantizerBundle(
            quant_outputs.quantization_loss, targets, codes, codes_pj
        )
    )
    return quantizers

  def apply_final_projection(self, x, quantizers):
    """Apply projection layer(s) on the features.

    A separate projection layer is used for each "section" (if using product
    quantization) of each quantizer.

    Args:
      x: Embeddings from late feature extractor of shape [bsz, sz, csz].
      quantizers: A list of QuantizerBundle's, one per quantizer.

    Returns:
      projected_x: A list whose length is the same as that of quantizers. Each
        element is the projected features of shape [ns, bsz, sz, final_dim].
    """
    projected_x = []
    # Create a separate (set of) projection(s) for each quantizer.
    for j in range(len(quantizers)):
      # A list of ns many elements that have shape [bsz, sz, csz/ns].
      x_sections = jnp.split(x, self.quantizer[j].get_num_sections(), axis=-1)
      # A list of ns many elements that have shape [bsz, sz, final_dim].
      x_proj = [
          nn.Dense(
              self.final_dim, name="final_proj_section_{}_quant_{}".format(i, j)
          )(x_sec)
          for (i, x_sec) in enumerate(x_sections)
      ]
      # [ns, bsz, sz, final_dim].
      projected_x.append(jnp.stack(x_proj, axis=0))
    return projected_x

  def get_logits(self, x_list, quantizers):
    """Compute the logits i.e.

    similarity between projected features and codes.

    Args:
      x_list: A list whose length is the number of quantizers. Each element of
        that list is an array of shape [ns, bsz, sz, final_dim], storing the
        features that were projected with a layer specific to that quantizer.
      quantizers: A list of the same length as x_list, storing the
        QuantizerBundle for each quantizers.

    Returns:
      The logits, as a list of [ns, bsz, sz, nc]-shaped jnp.arrays. The length
      of this list is the number of quantizers.
    """
    # Predict the code of each timestep using cosine similarity between the
    # projected embeddings and the projected codes.
    all_logits = []
    for x, q_bundle, q_module in zip(x_list, quantizers, self.quantizer):
      # First, l2-normalize the (projected) features and codes.
      x /= jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5
      codes_pj = q_bundle.projected_feature_codes
      codes_pj /= jnp.linalg.norm(codes_pj, axis=-1, keepdims=True) + 1e-5

      # Then, compute the dot product between them.
      ns = q_module.get_num_sections()
      codes_pj = jnp.transpose(codes_pj, (0, 2, 1))  # [ns, final_dim, nc]
      logits = jnp.dot(x, codes_pj)  # [ns, bsz, sz, ns, nc]
      # For each "section" of features, grab only the cluster assignments
      # corresponding to that section.
      logits = jnp.transpose(logits, (0, 3, 1, 2, 4))  # [ns, ns, bsz, sz, nc]
      # Out of the first 2 dims want to keep the inds [(0,0), (1,1), (2,2)...]
      inds = jnp.stack((jnp.arange(ns), jnp.arange(ns)), axis=1)
      logits = logits[tuple(jnp.moveaxis(inds, -1, 0))]  # [ns, bsz, sz, nc]

      # TODO(etriantafillou): experiment with learnable temperature.
      logits /= self.logit_temp
      all_logits.append(logits)
    return all_logits

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      train_mode_quantizer: bool,
      mask_key: jnp.ndarray | None,
  ) -> HubertOutput:
    """Apply the HuBERT model.

    The quantizer used may either be Product Quantizer (PQ) or a base quantizer.
    In the former case, instead of making a single centroid prediction per
    frame, a prediction is made for each "section" of the product quantizer.
    There is also a corresponding target for each section if using PQ, and the
    HuBERT loss becomes the average of the per-section losses.

    bsz: batch size.
    sz: number of frames (timesteps).
    csz: number of channels.
    nc: number of centroids.
    ns: number of sections of the product quantizer (if applicable).

    Args:
      inputs: Audio of shape `(bsz, sz)`.
      train: Whether we're in training mode (affects batch norm and dropout).
      train_mode_quantizer: Whether the quantizer is in train mode (affects EMA
        counter for cluster utilization).
      mask_key: A jnp.array that serves as the key for sampling masks. It can be
        None if `train` is False since no mask is applied in that case.

    Returns:
      Logits for which cluster each timestep belongs to (per section of the
        product quantizer, if applicable).
    """
    if len(self.quantizer) != len(self.quantizer_points):
      raise ValueError(
          "The lengths of `quantizer` and `quantizer_points` "
          "should match, but are {} and {}.".format(
              len(self.quantizer), len(self.quantizer_points)
          )
      )

    if self.omit_classifier_stop_grads is not None:
      for i in self.omit_classifier_stop_grads:
        if i < 0 or i >= len(self.readout_points):
          raise ValueError(
              "Requested to omit the stop-grad from classifier "
              f"with index {i} but there are only {len(self.readout_points)} "
              "readout points / classifiers."
          )

    model_outputs = {}
    quantizers = []

    if self.use_raw_audio:
      # Raw audio (no frontend) is fed through the early feature extractor.
      early_fs_inputs = jnp.expand_dims(inputs, -1)  # (bsz, sz, 1)
      x_earlyfs = self.early_feature_extractor(early_fs_inputs, train=train)
      if QuantizerPoints.FRONTEND.value in self.quantizer_points:
        x_frontend = self.frontend(inputs)  # (bsz, sz, csz)
        if x_earlyfs.shape[-2] != x_frontend.shape[-2]:
          raise ValueError(
              f"Expected the number of frontend frames ({x_frontend.shape[-2]})"
              " to match the number of frames from the early feature extractor "
              f"({x_earlyfs.shape[-2]}) in order to have as many HuBERT "
              "predictions as there are targets, since `quantizer_points` "
              "includes quantizing on top of the frontend."
          )
      x = x_earlyfs
    else:
      # Process audio with a frontend before the early feature extractor.
      x_frontend = self.frontend(inputs)  # (bsz, sz, csz)
      if self.early_feature_extractor is not None:
        x_earlyfs = self.early_feature_extractor(x_frontend, train=train)
      x = x_earlyfs

    # Add quantizers on frontend and/or early fs, if requested.
    if QuantizerPoints.FRONTEND.value in self.quantizer_points:
      quantizers = self.add_projected_quantizer(
          x_frontend, quantizers, train_mode_quantizer
      )

    if QuantizerPoints.EARLY_FS.value in self.quantizer_points:
      # Add the first quantizer, directly on top of the "early features".
      quantizers = self.add_projected_quantizer(
          x_earlyfs, quantizers, train_mode_quantizer
      )

    bsz, sz, csz = x.shape
    if self.add_positional_embeddings:
      x = x + conformer.PositionalEmbedding(embedding_dims=csz)(seq_length=sz)

    if self.stop_gradient_earlyfs:
      # If no early feature extractor is used, this should have no effect.
      # Otherwise, doing this will disallow HuBERT to train the early fs.
      # Note that this leads to not training the early fs at all (the quantizer
      # loss won't train it either, due to stopping gradients in quantizer.py).
      # Quantizing on top of random early features is maybe an interesting
      # baseline, if *consistency* of targets is what matters most.
      x = jax.lax.stop_gradient(x)

    # The learnable mask token.
    mask_emb = self.param("mask_emb", nn.initializers.uniform(), (csz,))

    # Get the corrupted x, where the features are replaced with the learnable
    # masked embedding for the positions that are chosen to be masked, if we are
    # in training mode.
    mask_idc = jnp.zeros((bsz, sz))
    if mask_key is not None:
      mask_idc = compute_mask_indices(
          mask_key, shape=(bsz, sz), **self.mask_config
      )
    model_outputs["mask_idc"] = mask_idc
    mask_idc_exp = jnp.repeat(jnp.expand_dims(mask_idc, 2), repeats=csz, axis=2)
    x = jnp.where(mask_idc_exp > 0, mask_emb, x)

    # Pass the corrupted x through the "late" feature extractor. Returns a list
    # of x's for the different "readout points".
    x_list = self.late_feature_extractor(
        x, train=train, return_intermediate_list=True
    )
    for block_ind in self.readout_points:
      if block_ind < 0 or block_ind >= len(x_list):
        raise ValueError(
            "Each element of `readout_points` should be in the "
            "range [0, len(x_list)) where x_list is the list that "
            "the late feature extractor returns. Found element "
            "{} and len(x_list) is {}".format(block_ind, len(x_list))
        )
    x = x_list[-1]  # the final-layer "embeddings"
    _, _, csz = x.shape
    model_outputs["embedding"] = x_list

    # Add additional quantizers on blocks of the late feature extractor.
    for point in list(self.quantizer_points):
      if (
          point == QuantizerPoints.FRONTEND.value
          or point == QuantizerPoints.EARLY_FS.value
      ):
        # Quantizers on the frontend and the early feature extractor will have
        # already been added, if requested. Nothing more to do here.
        continue
      elif point < 0:
        raise ValueError(
            "An element of `quantizer_points` can only be "
            f"negative if it's -1 or -2, but found {point}."
        )
      elif point >= len(x_list):
        raise ValueError(
            "Each element of `quantizer_points` should be in the "
            "range [0, len(x_list)) where x_list is the list that "
            "the late feature extractor returns. Found element "
            "{} and len(x_list) is {}".format(point, len(x_list))
        )
      quantizers = self.add_projected_quantizer(
          x_list[point], quantizers, train_mode_quantizer
      )

    # Linear readouts for supervised classification on top of HuBERT embeddings.
    classification_outputs = self.classify(
        x_list, mask_idc=mask_idc, **self.classifier_config
    )
    model_outputs.update(classification_outputs)

    # Final projection layer that projects embeddings to `final_dim`.
    # A list with as many elements as the number of quantizers used, where each
    # element has shape [ns, bsz, sz, final_dim].
    x_proj_list = self.apply_final_projection(x, quantizers)

    # Compute the logits via cosine similarity between the projected embeddings
    # and the projected codes.
    # A list of [ns, bsz, sz, nc]-shaped jnp.arrays with one item per quantizer.
    logits = self.get_logits(x_proj_list, quantizers)
    model_outputs["logits"] = logits

    # The targets for each quantizer.
    model_outputs["targets"] = [
        quantizers[i].targets for i in range(len(quantizers))
    ]

    # The quantization loss: the mean over the individual quantizer losses.
    # [bsz, sz, nc].
    quant_losses = [
        quantizers[i].quantization_loss for i in range(len(quantizers))
    ]
    model_outputs["quantization_loss"] = jnp.mean(
        jnp.stack(quant_losses, axis=0), axis=0
    )

    return HubertOutput(**model_outputs)  # pytype: disable=wrong-arg-types  # jnp-type
