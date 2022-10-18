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

"""Entry script for source-free domain adaptation."""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from chirp import config_utils
from chirp.projects.sfda import adapt
from chirp.projects.sfda import data_utils
from chirp.projects.sfda.configs import config_globals
import jax
from ml_collections.config_flags import config_flags
import tensorflow as tf

_CONFIG = config_flags.DEFINE_config_file("config")
_METHOD_CONFIG = config_flags.DEFINE_config_file(
    "method_config",
    help_string="Configuration file for method-specific hyperparamaters.")
_LOGDIR = flags.DEFINE_string("logdir", None, "Work unit logging directory.")
flags.mark_flags_as_required(["config", "logdir"])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  logging.info(_CONFIG.value)
  logging.info(_METHOD_CONFIG.value)
  # Preventing tensorflow from taking any GPU memory and starving Jax.
  tf.config.experimental.set_visible_devices([], "GPU")
  config = config_utils.parse_config(_CONFIG.value,
                                     config_globals.get_globals())
  method_config = config_utils.parse_config(_METHOD_CONFIG.value,
                                            config_globals.get_globals())

  if jax.local_device_count() > 1:
    raise NotImplementedError(
        "Only supporting non-distributed setting for now.")
  # Recover the SDFA method
  sfda_method = method_config.sfda_method

  # Convert the splits
  config = config.unlock()
  method_config = getattr(method_config, config.modality.value)

  if config.modality == adapt.Modality.AUDIO:
    adaptation_dataset, val_dataset = data_utils.get_audio_datasets(
        adaptation_data_config=config.adaptation_data_config,
        eval_data_config=config.eval_data_config,
        sample_rate_hz=config.sample_rate_hz)
  else:
    adaptation_dataset, val_dataset = data_utils.get_image_datasets(
        image_model=config.model_config.encoder,
        dataset_name=config.init_config.target_class_list,
        batch_size_train=config.batch_size_adaptation,
        batch_size_eval=config.batch_size_eval,
        data_seed=config.init_config.rng_seed)

  # Initialize state and bundles
  model_bundle, adaptation_state, key = sfda_method.initialize(
      model_config=config.model_config,
      pretrained=config.init_config.pretrained_model,
      rng_seed=config.init_config.rng_seed,
      input_shape=None if config.modality == adapt.Modality.IMAGE else
      config.init_config.input_shape,
      target_class_list=config.init_config.target_class_list,
      adaptation_iterations=len(adaptation_dataset) * method_config.num_epochs,
      modality=config.modality,
      optimizer_config=method_config.optimizer_config)

  # Perform adaptation
  adaptation_state = adapt.perform_adaptation(
      key=key,
      adaptation_state=adaptation_state,
      adaptation_dataset=adaptation_dataset,
      validation_dataset=val_dataset,
      model_bundle=model_bundle,
      logdir=_LOGDIR.value,
      use_supervised_metrics=True,
      target_class_list=config.init_config.target_class_list,
      multi_label=config.multi_label,
      modality=config.modality,
      eval_every=config.eval_every,
      sfda_method=sfda_method,
      **method_config,
  )


if __name__ == "__main__":
  app.run(main)
