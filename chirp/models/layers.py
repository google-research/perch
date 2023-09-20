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

"""Model layers.

Building blocks and layers to construct networks, implemented as Flax modules.

"""
from typing import Callable

from flax import linen as nn
import jax
from jax import nn as jnn
from jax import numpy as jnp
import optax

JTensor = jnp.ndarray


class SqueezeAndExcitation(nn.Module):
  """Squeeze-and-Excitation layer.

  See "Squeeze-and-Excitation Networks" (Hu et al., 2018), particularly
  equations 2 and 3.

  Attributes:
    reduction_ratio: The reduction factor in the squeeze operation. Referred to
      as `r` in the paper.
    activation: The activation to apply after squeezing.
  """

  reduction_ratio: int = 4
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies SqueezeAndExcite on the 2D inputs.

    Args:
      inputs: Input data in shape of `(batch size, height, width, channels)`.

    Returns:
      JAX array with same shape as the input.
    """
    if inputs.ndim != 4:
      raise ValueError(
          "Inputs should in shape of `[batch size, height, width, features]`"
      )

    # Squeeze
    x = jnp.mean(inputs, axis=(1, 2))
    x = nn.Dense(features=x.shape[-1] // self.reduction_ratio, name="Reduce")(x)
    x = self.activation(x)

    # Excite
    x = nn.Dense(features=inputs.shape[-1], name="Expand")(x)
    x = nn.sigmoid(x)
    return inputs * x[:, None, None, :]


class MBConv(nn.Module):
  """Mobile inverted bottleneck block.

  As introduced in "Mobilenetv2: Inverted residuals and linear bottlenecks"
  (Sandler et al., 2018). See figure 4d for an illustration and
  https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet for
  a reference implementation.

  The defaults are those from the MobileNetV2 paper. There is added support for
  batch normalization and squeeze-and-excitation blocks as used by EfficientNet.
  Note that the skip connection is not part of this module.

  Attributes:
    features: The number of filters.
    strides: The strides to use in the depthwise separable convolution.
    expand_ratio: The expansion factor to use. A block with expansion factor `N`
      is commonly referred to as MBConvN.
    kernel_size: The kernel size used by the depthwise separable convolution.
    activation: The activation function to use after the expanding 1x1
      convolution. Also used by the optional squeeze-and-excitation block.
    batch_norm: Whether to use batch normalization after the expanding and
      reducing convolutions.
    reduction_ratio: If given, a squeeze-and-excitation block is inserted after
      the depthwise separable convolution with the given reduction factor. Note
      that this reduction ratio is relative to the number of input channels,
      i.e., it scales with `expand_ratio`.
  """

  features: int
  strides: int
  expand_ratio: int
  kernel_size: tuple[int, int] = (3, 3)
  activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.relu6
  batch_norm: bool = False
  reduction_ratio: int | None = None

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, use_running_average: bool = None
  ) -> jnp.ndarray:
    """Applies an inverted bottleneck block to the inputs.

    Args:
      inputs: Inputs should be of shape `(batch size, height, width, channels)`.
      use_running_average: Used to decide whether to use running statistics in
        BatchNorm (test mode), or the current batch's statistics (train mode).

    Returns:
      A JAX array of `(batch size, height, width, features)`.
    """
    features = self.expand_ratio * inputs.shape[-1]

    x = inputs
    if self.expand_ratio != 1:
      x = nn.Conv(
          features=features,
          kernel_size=(1, 1),
          strides=(1, 1),
          use_bias=False,
          name="ExpandConv",
      )(x)
      if self.batch_norm:
        x = nn.BatchNorm(
            use_running_average=use_running_average, name="ExpandBatchNorm"
        )(x)
      x = self.activation(x)

    if self.strides == 2:

      def _pad_width(input_size: int, kernel_size: int) -> tuple[int, int]:
        """Calculate padding required to halve input with stride 2."""
        return (kernel_size // 2) - (1 - input_size % 2), kernel_size // 2

      padding = (
          _pad_width(x.shape[1], self.kernel_size[0]),
          _pad_width(x.shape[2], self.kernel_size[1]),
      )
    else:
      padding = "SAME"

    x = nn.Conv(
        features=features,
        kernel_size=self.kernel_size,
        strides=self.strides,
        padding=padding,
        feature_group_count=features,
        use_bias=False,
        name="DepthwiseConv",
    )(x)
    if self.batch_norm:
      x = nn.BatchNorm(
          use_running_average=use_running_average, name="DepthwiseBatchNorm"
      )(x)
    x = self.activation(x)

    if self.reduction_ratio is not None:
      x = SqueezeAndExcitation(
          reduction_ratio=self.reduction_ratio * self.expand_ratio,
          activation=self.activation,
      )(x)
    x = nn.Conv(
        features=self.features,
        kernel_size=(1, 1),
        strides=1,
        use_bias=False,
        name="ProjectConv",
    )(x)
    if self.batch_norm:
      x = nn.BatchNorm(
          use_running_average=use_running_average, name="ProjectBatchNorm"
      )(x)

    return x


class Identity(nn.Module):
  """Identity layer."""

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    """Identity function.

    Args:
      inputs: Input array.
      *args: Any other arguments are ignored.
      **kwargs: Any keyword arguments are ignored.

    Returns:
      The input, unchanged.
    """
    return inputs


class FeedForward(nn.Module):
  """Linear layer.

  Attributes:
    output_dims: Depth of the output.
    activation: The activation to apply after the linear layer.
  """

  output_dims: int = 0
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies a feed forward layer to inputs.

    Args:
      inputs: The inputs jnp.ndarray.  Shaped [..., input_dims].

    Returns:
      Outputs. Shaped [..., output_dims].
    """
    x = nn.Dense(features=self.output_dims, name="FeedForward")(inputs)
    x = self.activation(x)
    return x


# Transformer layers.
class TransformerFeedForward(nn.Module):
  """Transformer feedforward layer with residual connection and dropout.

  Attributes:
    input_dims: Depth of the input.
    hidden_dims: Hidden dimension of FFN.
    activation: Activation function to use. Options are RELU, RELU6, RELU^2,
      RELU^3, SIGMOID, TANH, GELU, GATED_GELU, GATED_SILU, NONE.
    residual_dropout_prob: Residual dropout.
    relu_dropout_prob: FFN dropout.
    add_skip_connection: Whether to add residual connection.
    residual_weight: Weight of the residual connection. Output = fn(x) *
      residual_weight + x.
  """

  input_dims: int = 0
  hidden_dims: int = 0
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  residual_dropout_prob: float = 0.0
  relu_dropout_prob: float = 0.0
  add_skip_connection: bool = True
  residual_weight: float = 1.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    output_dims = self.input_dims
    inputs_normalized = nn.LayerNorm(name="layer_norm")(inputs)

    # Apply first FFN layer
    projected_inputs = FeedForward(
        output_dims=self.hidden_dims, activation=self.activation
    )(inputs_normalized)

    # Apply RELU dropout
    projected_inputs = nn.Dropout(self.relu_dropout_prob)(
        projected_inputs, deterministic=not train
    )

    # Apply second FFN layer
    projected_inputs = FeedForward(
        output_dims=output_dims, activation=Identity()
    )(projected_inputs)

    # Apply residual dropout
    projected_inputs = nn.Dropout(self.residual_dropout_prob)(
        projected_inputs, deterministic=not train
    )

    # Apply skip connection
    if self.add_skip_connection:
      projected_inputs = inputs + projected_inputs * self.residual_weight

    return projected_inputs


# Convolution layers.
class LightConv1D(nn.Module):
  """Lightweight conv layer.

  architecture::

  input-ln()-ff()-glu()-depthwise_conv1d()-norm()-act()-ff()-dropout()-+-output
    |__________________________________________________________________|

  Attributes:
    input_dims:      Input and (in fact,) output dimension.
    kernel_size:     Kernel size of 1d deptwise conv.
    conv_activation: Activation after normalization.
    dropout_prob:    Dropout probability.
  """

  input_dims: int | None = None
  kernel_size: int | None = None
  conv_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
  dropout_prob: float = 0.0
  downsample: bool = True

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      use_running_average: bool | None = None,
  ) -> jnp.ndarray:
    """Lightweight conv layer.

    Args:
      inputs: Input sequence jnp.ndarray of shape [B, T, H].
      train: Whether this is training. This affects Dropout behavior, and also
        affects BatchNorm behavior if 'use_running_average' is set to None.
      use_running_average: Optional, used to decide whether to use running
        statistics in BatchNorm (test mode), or the current batch's statistics
        (train mode). If not specified (or specified to None), default to 'not
        train'.

    Returns:
      The lconv output with shape [B, T, H].
    """
    if use_running_average is None:
      use_running_average = not train
    unnormalized_inputs = inputs

    inputs = nn.LayerNorm(name="ln")(inputs)
    act_inputs = FeedForward(
        output_dims=self.input_dims, activation=Identity()
    )(inputs)
    gated_inputs = FeedForward(
        output_dims=self.input_dims, activation=Identity()
    )(inputs)
    inputs = act_inputs * jax.nn.sigmoid(gated_inputs)

    inputs = nn.Conv(
        features=self.input_dims,
        kernel_size=(self.kernel_size,),
        strides=2 if self.downsample else 1,
        padding="SAME",
        input_dilation=1,
        kernel_dilation=1,
        feature_group_count=self.input_dims,
        use_bias=False,
    )(inputs)

    inputs = nn.BatchNorm()(inputs, use_running_average=use_running_average)
    inputs = self.conv_activation(inputs)

    inputs = FeedForward(output_dims=self.input_dims, activation=Identity())(
        inputs
    )
    inputs = nn.Dropout(self.dropout_prob)(inputs, deterministic=not train)

    if self.downsample:
      unnormalized_inputs = nn.avg_pool(
          unnormalized_inputs, (2,), (2,), padding="SAME"
      )
      # If downsampling happened, the dimensions might also have changed, which
      # means we need to project the inputs for the residual connection
      if unnormalized_inputs.shape[-1] != self.input_dims:
        unnormalized_inputs = nn.Dense(features=self.input_dims)(
            unnormalized_inputs
        )

    output = inputs + unnormalized_inputs
    return output


# Conformer layers.
class SelfAttentionWithNormAndResidual(nn.Module):
  """Self attention sub-layer used in the Conformer layer.

  Input is first normalized using layer norm. Output is processed using
  multi-headed attention. And finally, the output of the attention layer
  is combined with the input by residual connection.

  For the normalization, we can specify pre norm or post norm.
  For the residual connection, we can specify the residual weight.

  Attributes:
    residual_weight: Weight of the residual connection. Output = fn(x) *
      residual_weight + x * input_weight.
    input_weight: Weight of the input connection. Output = fn(x) *
      residual_weight + x * input_weight.
    pre_layer_norm: Whether to apply norm before or after the layer.
    residual_dropout_prob: Probability at which we apply dropout to the residual
      layers, such that, residual(x, y) = (x + dropout(y)).
  """

  residual_weight: float = 1.0
  input_weight: float = 1.0
  pre_layer_norm: bool = True
  residual_dropout_prob: float = 0.0
  atten_dropout_prob: float = 0.0
  num_heads: int = 1

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      atten_mask: JTensor | None = None,
  ) -> jnp.ndarray:
    unnormalized_inputs = inputs

    if self.pre_layer_norm:
      inputs = nn.LayerNorm()(inputs)

    self_atten = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads, dropout_rate=self.atten_dropout_prob
    )
    result = self_atten(
        inputs_q=inputs,
        inputs_kv=inputs,
        mask=atten_mask,
        deterministic=not train,
    )

    if not self.pre_layer_norm:
      result = nn.LayerNorm()(result)

    dropout = nn.Dropout(self.residual_dropout_prob, name="residual_dropout")
    result = (
        dropout(result, deterministic=not train) * self.residual_weight
        + unnormalized_inputs * self.input_weight
    )
    return result


class Conformer(nn.Module):
  """Conformer layer as in https://arxiv.org/abs/2005.08100.

  Canonical version (with default params.)
    x = x + 1/2 * FFN(x)
    x = x + MHSA(x)
    x = x + Lconv(x)
    x = x + 1/2 * FFN(x)
    y = ln(x)

  Residual connections are implemented inside each individual block:
    FFN, MHSA, LConv.
  Optionally one can change the order of MHSA and conv.

  Attributes:
    model_dims: Encoder model dimension.
    kernel_size: Conv kernel size.
    ff_activation: Activation function used in the feedforward network.
    ff_residual_weight: Residual weight used in the fflayer.
    ffn_dim_multiplier: Feed forward hidden dimension will be ffn_dim_multiplier
      * model_dims.
    atten_num_heads: Number of attention heads.
    layer_order: Only mhsa, conv, mhsa_before_conv or conv_before_mhsa are
      supported
    dropout_prob: Dropout prob of inner components.
    conv_residual_dropout: Conv block residual dropout. Will be overwritten by
      p.dropout if it is not None.
    atten_residual_dropout: Attention block residual dropout. Will be
      overwritten by p.dropout if it is not None.
    ffn_residual_dropout: Feed forward block residual dropout. Will be
      overwritten by p.dropout if it is not None.
    atten_dropout: Dropout in Attention layer. Will be overwritten by p.dropout
      if it is not None.
    ffn_relu_dropout: Post activation dropout in Feed-forward layer. Will be
      overwritten by p.dropout if it is not None.
    fflayer_weight_sharing: If True, will ignore `fflayer_end_tpl`, and will
      make the fflayer_end layer as a weight-shared copy of the fflayer_start
      layer.
  """

  model_dims: int = 512
  kernel_size: int = 32
  ff_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
  ff_residual_weight: float = 0.5
  ffn_dim_multiplier: int = 4
  atten_num_heads: int = 8
  layer_order: str = "mhsa_before_conv"
  dropout_prob: float | None = None
  conv_residual_dropout: float | None = None
  atten_residual_dropout: float | None = None
  ffn_residual_dropout: float | None = None
  atten_dropout: float | None = None
  ffn_relu_dropout: float | None = None
  fflayer_weight_sharing: bool = False
  downsample: bool = False
  skip_layer_norm: bool = True

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      use_running_average: bool | None = None,
      atten_mask: jnp.ndarray | None = None,
  ) -> jnp.ndarray:
    """Conformer layer.

    Args:
      inputs: Input sequence jnp.ndarray of shape [B, T, H].
      train: Whether this is training. This affects Dropout behavior, and also
        affects BatchNorm behavior if 'use_running_average' is set to None.
      use_running_average: Optional, used to decide whether to use running
        statistics in BatchNorm (test mode), or the current batch's statistics
        (train mode). If not specified (or specified to None), default to 'not
        train'.
      atten_mask: Input jnp.ndarray attention mask.

    Raises:
      RuntimeError: if an attention mask is given but there's no attention layer

    Returns:
      The conformer output with shape [B, T, D].
    """
    if use_running_average is None:
      use_running_average = not train

    layer_order_set = ["mhsa", "conv", "mhsa_before_conv", "conv_before_mhsa"]
    if self.layer_order not in layer_order_set:
      raise ValueError(
          f"`self.layer_order` must be within `{layer_order_set}`."
      )

    input_dims = inputs.shape[-1]

    # Set up the first ff layer.
    fflayer_start = TransformerFeedForward(
        name="fflayer_start",
        activation=self.ff_activation,
        input_dims=input_dims,
        hidden_dims=input_dims * self.ffn_dim_multiplier,
        residual_weight=self.ff_residual_weight,
        residual_dropout_prob=self.ffn_residual_dropout,
        relu_dropout_prob=self.ffn_relu_dropout,
    )

    # Set up the last ff layer.
    fflayer_end = TransformerFeedForward(
        name="fflayer_end",
        activation=self.ff_activation,
        input_dims=self.model_dims,
        hidden_dims=self.model_dims * self.ffn_dim_multiplier,
        residual_weight=self.ff_residual_weight,
        residual_dropout_prob=self.ffn_residual_dropout,
        relu_dropout_prob=self.ffn_relu_dropout,
    )

    # Setup attention layer.
    if "mhsa" in self.layer_order:
      trans_atten = SelfAttentionWithNormAndResidual(
          residual_dropout_prob=self.atten_residual_dropout,
          atten_dropout_prob=self.atten_dropout,
          num_heads=self.atten_num_heads,
      )

    # Setup convolution layer.
    lconv = LightConv1D(
        input_dims=self.model_dims,
        kernel_size=self.kernel_size,
        dropout_prob=self.conv_residual_dropout,
        downsample=self.downsample,
    )

    if not self.skip_layer_norm:
      final_ln = nn.LayerNorm(name="final_ln")

    if atten_mask is not None and "mhsa" not in self.layer_order:
      raise RuntimeError("Attention mask is provided but no attention layer.")

    inputs = fflayer_start(inputs, train)

    if self.layer_order == "mhsa":
      inputs = trans_atten(inputs=inputs, train=train, atten_mask=atten_mask)
    elif self.layer_order == "conv":
      inputs = lconv(
          inputs, train=train, use_running_average=use_running_average
      )
    elif self.layer_order == "mhsa_before_conv":
      inputs = trans_atten(inputs=inputs, train=train, atten_mask=atten_mask)
      inputs = lconv(inputs, train)
    else:
      inputs = lconv(inputs, train)
      inputs = trans_atten(inputs=inputs, train=train, atten_mask=atten_mask)

    if self.fflayer_weight_sharing:
      # With the weight sharing, we apply fflayer_start again
      inputs = fflayer_start(inputs, train)
    else:
      inputs = fflayer_end(inputs, train)

    if not self.skip_layer_norm:
      inputs = final_ln(inputs)
    return inputs


class StridedAutopool(nn.Module):
  """Strided 1D Autopool over an array of shape [B, T, D].

  See https://arxiv.org/abs/1804.10070 for basic Autopool derivation.
  This implementation applies autopool to strided time windows.
  """

  alpha_0: float
  pool_width: int
  pool_stride: int
  padding: str

  @nn.compact
  def __call__(self, inputs):
    alpha_shape = [1] * (len(inputs.shape) - 1) + [inputs.shape[-1]]
    alpha = self.param(
        "alpha", nn.initializers.constant(self.alpha_0), alpha_shape
    )

    pool_fn = lambda x: nn.pooling.avg_pool(  # pylint: disable=g-long-lambda
        x,
        window_shape=(self.pool_width,),
        strides=(self.pool_stride,),
        padding=self.padding,
    )
    exp_inputs = jnp.exp(alpha * inputs)
    auto_pooled = pool_fn(exp_inputs * inputs) / pool_fn(exp_inputs)
    return auto_pooled


class EarlyFeatureExtractor(nn.Module):
  """Network used as the "early feature extractor" for HuBERT.

  This module is comprised of a number of convolutional layers. It also uses
  group normalization after the first layer only. It is based on the
  architecture used for wav2vec 2.0 / HuBERT, and using the defaults of the
  implementation from
  https://github.com/facebookresearch/fairseq/blob/5307a0e078d7460003a86f4e2246d459d4706a1d/fairseq/models/wav2vec/wav2vec2.py

    Attributes:
      conv_layer_tuples: A List of (dim, kernel size, stride) tuples, one for
        each of the convolutional layers.
      dropout_prob: A float. The dropout probability.
      activation: The activation to apply after each convolutional "block".
      deprecated_group_conv: Whether to use the older version of this layer
        (which used grouped convolutions), for compatibility with old
        experiments. This option will be removed in the future.
  """

  conv_layer_tuples: tuple[tuple[int, int, int], ...]
  dropout_prob: float = 0.0
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  deprecated_group_conv: bool = False

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    """Convolutional feature extractor used for "early" feature extraction.

    Args:
      inputs: Input sequence jnp.ndarray of shape [B, T, H].
      train: Whether we are in training mode. Affects dropout.

    Returns:
      A jnp.ndarray with shape [B, T, D].
    """
    if self.deprecated_group_conv:
      if inputs.ndim != 3:
        raise ValueError("Expected the input to have 3 dimensions.")
      model_dims = self.conv_layer_tuples[0][0]
      if inputs.shape[-1] != model_dims:
        inputs = FeedForward(output_dims=model_dims)(inputs)

    # TODO(etriantafillou): Experiment with adding residual connections.
    for i, (dim, k, stride) in enumerate(self.conv_layer_tuples):
      inputs = nn.Conv(
          features=dim,
          kernel_size=(k,),
          strides=(stride,),
          feature_group_count=dim if self.deprecated_group_conv else 1,
          use_bias=False,
          name="conv_layer_{}".format(i),
      )(inputs)

      inputs = nn.Dropout(self.dropout_prob)(inputs, deterministic=not train)

      if i == 0:
        if self.deprecated_group_conv:
          inputs = nn.GroupNorm(num_groups=None, group_size=dim)(inputs)
        else:
          inputs = nn.GroupNorm(num_groups=dim)(inputs)

      inputs = self.activation(inputs)

    return inputs


def hinge_loss(predictor_outputs, targets):
  """Computes the hinge loss while accommodating targets in {0, 1}."""
  targets = 2 * targets - 1
  return optax.hinge_loss(predictor_outputs, targets)
