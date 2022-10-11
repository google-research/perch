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

from typing import Optional, Tuple

from chirp import train
from chirp.models import taxonomy_model
from chirp.taxonomy import class_utils
import flax
from flax.core import scope
import flax.linen as nn
import jax.numpy as jnp
from ml_collections import config_dict
import optax


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
    optimizer_config: config_dict.ConfigDict,
    total_steps: int,
    rng_seed: int,
    input_shape: Tuple[int, ...],
    pretrained_ckpt_dir: str,
    target_class_list: str,
) -> Tuple[ModelBundle, scope.VariableDict, scope.FrozenVariableDict,
           scope.FrozenVariableDict]:
  """Loads the taxonomic classifier's and optimizer's params and states.

  Args:
    model_config: The model configuration, including the definitions of the
      different parts of the architecture.
    optimizer_config: The optimizer configuration, including the name of the
      optimizer, the learning rate etc.
    total_steps: The total number of steps used for adaptation. Used to
      adequately define learning rate scheduling.
    rng_seed: The random seed used to initialize the model.
    input_shape: The shape of the input (for audio, equals to [sample_rate_hz *
      audio_length_s]).
    pretrained_ckpt_dir: The directory where to find the pretrained checkpoint.
    target_class_list: The classlist in which labels are expressed. Used to
      define the size of the classifier's head.

  Returns:
    model_bundle: The ModelBundle, include the taxonomic model and its
      optimizer.
    params: The model's params after loading.
    model_state: The model' state.
    opt_state: The optimizer's state.
  """

  # Load main classification model from pretrained checkpoint
  model_bundle, train_state = train.initialize_model(
      model_config=model_config,
      rng_seed=rng_seed,
      input_shape=input_shape,
      learning_rate=0.,
      workdir=pretrained_ckpt_dir,
      target_class_list=target_class_list)

  class_lists = class_utils.get_class_lists(
      target_class_list, add_taxonomic_labels=False)
  num_classes = {k: v.size for (k, v) in class_lists.items()}
  model = taxonomy_model.TaxonomyModel(
      num_classes=num_classes,
      encoder=model_config.encoder,
      frontend=model_config.frontend,
      taxonomy_loss_weight=0.)

  # Define the optimizer
  params = train_state.params
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
          train.project(0.0, jnp.pi), train.mask_by_name("gabor_mean", params)),
      optax.masked(
          train.project(0.0, jnp.pi), train.mask_by_name("gabor_mean", params)),
      optax.masked(
          train.project(4 * std_to_fwhm,
                        model_bundle.model.frontend.kernel_size * std_to_fwhm),
          train.mask_by_name("gabor_std", params)),
  )
  opt_state = optimizer.init(params)
  model_bundle = ModelBundle(model, optimizer)
  return model_bundle, params, train_state.model_state, opt_state
