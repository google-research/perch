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

"""Utilities to prepare models for SFDA methods."""

import enum
from typing import Optional, Tuple, Union

from absl import logging
from chirp import train
from chirp.projects.sfda import data_utils
from chirp.projects.sfda import models
from chirp.projects.sfda.models import image_model
from chirp.projects.sfda.models import taxonomy_model
from chirp.taxonomy import class_utils
import flax
from flax.core import scope
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import optax


class TrainableParams(enum.Enum):
  """Defines which set of trainable parameters will be adapted.

  Attributes:
    ALL: All parameters will be adapted.
    BN: Only BatchNorm scale and bias trainable parameters will be adapted.
      Whether the population statistics will be adapted or not is orthogonal,
      and controlled by the 'update_bn_statistics' option, in each method's
      config file.
  """
  ALL = "all"
  BN = "batch_norm"


def mask_parameters(params: flax.core.scope.VariableDict,
                    strategy: TrainableParams,
                    model: Union[image_model.ImageModel,
                                 taxonomy_model.TaxonomyModel]):
  """Creates the mask of parameters to which zero_grad() will be applied.

  Args:
    params: The Pytree representing all of the model's parameters.
    strategy: The strategy to use for masking.
    model: The model considered. Used to determine whether some parameter
      belongs a BatchNorm layer or not.

  Returns:
    unflattened_mask: A mask with the same Tree structure as Pytree, but
    with boolean leaves indicating whether some parameter should be masked or
    not.
  """
  flat_tree = flax.traverse_util.flatten_dict(params)
  if strategy == TrainableParams.BN:
    mask = {k: not model.is_bn_parameter(k) for k in flat_tree}
  elif strategy == TrainableParams.ALL:
    mask = {k: False for k in flat_tree}
  else:
    raise NotImplementedError(f"Strategy {strategy} is not supported yet.")
  frozen_parameters = [p for p, masked in mask.items() if masked]
  trainable_parameters = [p for p, masked in mask.items() if not masked]
  logging.info(
      "The following parameters will be kept frozen during adaptation:"
      " %s", frozen_parameters)
  logging.info(
      "The following parameters will be trained during adaptation:"
      " %s", trainable_parameters)
  return flax.traverse_util.unflatten_dict(mask)


@flax.struct.dataclass
class ModelBundle:
  """Model and optimizer definition.

  Attributes:
    model: The model used for adaptation.
    optimizer: The optimizer used for adaptation.
  """
  model: nn.Module
  optimizer: Optional[optax.GradientTransformation]


def prepare_audio_model(
    model_config: config_dict.ConfigDict,
    optimizer_config: Optional[config_dict.ConfigDict],
    pretrained: bool,
    total_steps: int,
    rng_seed: int,
    input_shape: Tuple[int, ...],
    target_class_list: str,
) -> Tuple[ModelBundle, scope.VariableDict, scope.FrozenVariableDict,
           Optional[scope.FrozenVariableDict]]:
  """Loads the taxonomic classifier's and optimizer's params and states.

  Args:
    model_config: The model configuration, including the definitions of the
      different parts of the architecture.
    optimizer_config: The optimizer configuration, including the name of the
      optimizer, the learning rate etc. If set to None, the returned ModelBundle
      will contain None in place of the optimizer, and the returned opt_state
      will be None.
    pretrained: Whether to load the pretrained model. If set to True,
      model_config.pretrained_ckpt_dir will be used to load the model.
    total_steps: The total number of steps used for adaptation. Used to
      adequately define learning rate scheduling.
    rng_seed: The random seed used to initialize the model.
    input_shape: The shape of the input (for audio, equals to [sample_rate_hz *
      audio_length_s]).
    target_class_list: The classlist in which labels are expressed. Used to
      define the size of the classifier's head.

  Returns:
    model_bundle: The ModelBundle, include the taxonomic model and its
      optimizer.
    params: The model's params after loading.
    model_state: The model' state.
    opt_state: The optimizer's state.
  """
  # Define the main model
  class_lists = class_utils.get_class_lists(
      target_class_list, add_taxonomic_labels=False)
  num_classes = {k: v.size for (k, v) in class_lists.items()}
  model = taxonomy_model.TaxonomyModel(
      num_classes=num_classes,
      encoder=model_config.encoder,
      frontend=model_config.frontend,
      taxonomy_loss_weight=0.)

  if pretrained:
    # Load main classification model from pretrained checkpoint
    ckpt_dir = model_config.pretrained_ckpt_dir
    # 'pretrained_ckpt_dir' interferes with train.initialize_model, as the
    # creation of a TaxonomyModel does not expect this argument. Therefore,
    # we delete it here to ensure compatibility.
    delattr(model_config, "pretrained_ckpt_dir")
    model_bundle, train_state = train.initialize_model(
        model_config=model_config,
        rng_seed=rng_seed,
        input_shape=input_shape,
        learning_rate=0.,
        workdir=ckpt_dir,
        target_class_list=target_class_list)
    train_state = model_bundle.ckpt.restore(train_state)
    params = train_state.params
    model_state = train_state.model_state
  else:
    variables = model.init(
        jax.random.PRNGKey(rng_seed),
        jnp.zeros((1,) + input_shape),
        train=False)
    model_state, params = variables.pop("params")
    params = params.unfreeze()
  # Define the optimizer
  if optimizer_config is None:
    optimizer = None
    opt_state = None
  else:
    std_to_fwhm = jnp.sqrt(2 * jnp.log(2)) / jnp.pi
    if optimizer_config.use_cosine_decay:
      print(f"Using cosine decay with {total_steps} steps.")
      learning_rate = optax.cosine_decay_schedule(
          optimizer_config.learning_rate, decay_steps=total_steps)
    else:
      learning_rate = optimizer_config.learning_rate
    opt = getattr(optax, optimizer_config.optimizer)
    optimizer = optax.chain(
        opt(learning_rate=learning_rate, **optimizer_config.opt_kwargs),
        optax.masked(
            train.project(0.0, 1.0),
            train.mask_by_name("spcen_smoothing_coef", params)),
        optax.masked(
            train.project(0.0, jnp.pi),
            train.mask_by_name("gabor_mean", params)),
        optax.masked(
            train.project(0.0, jnp.pi),
            train.mask_by_name("gabor_mean", params)),
        optax.masked(
            train.project(4 * std_to_fwhm,
                          model.frontend.kernel_size * std_to_fwhm),
            train.mask_by_name("gabor_std", params)),
        optax.masked(
            zero_grads(),
            mask_parameters(params, optimizer_config.trainable_params_strategy,
                            model)),
    )
    opt_state = optimizer.init(params)
  model_bundle = ModelBundle(model, optimizer)
  return model_bundle, params, model_state, opt_state


def prepare_image_model(
    model_config: config_dict.ConfigDict,
    optimizer_config: Optional[config_dict.ConfigDict], total_steps: int,
    rng_seed: int, pretrained: bool, target_class_list: str, **_
) -> Tuple[ModelBundle, scope.VariableDict, scope.FrozenVariableDict,
           Optional[scope.FrozenVariableDict]]:
  """Prepare an image model for source-free domain adaptation.

  Args:
    model_config: The model configuration, including the specification of the
      encoder's architecture.
    optimizer_config: The optimizer configuration, including the name of the
      optimizer, the learning rate etc.
    total_steps: The total number of steps used for adaptation. Used to
      adequately define learning rate scheduling.
    rng_seed: The seed to initialize the model, in case no pretrained checkpoint
      is provided.
    pretrained: Whether to load the model from a pretrained checkpoint or not.
      If set to True, the model will use the 'load_ckpt' method from the
      corresponding model.
    target_class_list: The name of the dataset used for adaptation. This is used
      to grab the correct checkpoint for each model.

  Returns:
      model_bundle: The ModelBundle, including the image model and its
        optimizer.
      params: The model's params after loading.
      model_state: The model' state.
      opt_state: The optimizer's state.
  """
  data_info = data_utils.get_metadata(target_class_list)
  model = models.MODEL_REGISTRY[model_config.encoder](
      num_classes=data_info["num_classes"])
  if pretrained:
    variables = model.load_ckpt(target_class_list)
  else:
    input_shape = (data_info["resolution"], data_info["resolution"], 3)
    variables = model.init(
        jax.random.PRNGKey(rng_seed), jnp.zeros((1,) + input_shape), False,
        False)
  model_state, params = variables.pop("params")
  params = params.unfreeze()
  if optimizer_config is None:
    optimizer = None
    opt_state = None
  else:
    if optimizer_config.use_cosine_decay:
      learning_rate = optax.cosine_decay_schedule(
          optimizer_config.learning_rate, decay_steps=total_steps)
    else:
      learning_rate = optimizer_config.learning_rate
    opt = getattr(optax, optimizer_config.optimizer)

    optimizer = optax.chain(
        opt(learning_rate=learning_rate, **optimizer_config.opt_kwargs),
        optax.masked(
            zero_grads(),
            mask_parameters(params, optimizer_config.trainable_params_strategy,
                            model)))
    opt_state = optimizer.init(params)
  model_bundle = ModelBundle(model, optimizer)
  return model_bundle, params, model_state, opt_state


def zero_grads() -> optax.GradientTransformation:
  """Creates a GradientTransformation that zeros out gradients."""

  def init_fn(_):
    return ()

  def update_fn(updates, state, params=None):  # pylint: disable=unused-argument
    return jax.tree_map(jnp.zeros_like, updates), ()

  return optax.GradientTransformation(init_fn, update_fn)
