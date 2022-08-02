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

"""Adapt a taxonomy classifier in a source-free fashion."""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from chirp import adapt
from chirp import config_utils
from chirp.configs import config_globals
from chirp.data import pipeline
from ml_collections.config_flags import config_flags
import tensorflow as tf

from xmanager import xm  # pylint: disable=unused-import

_CONFIG = config_flags.DEFINE_config_file("config")
_METHOD_CONFIG = config_flags.DEFINE_config_file("method_config")
_LOGDIR = flags.DEFINE_string("logdir", None, "Work unit logging directory.")
_WORKDIR = flags.DEFINE_string("workdir", None,
                               "Work unit checkpointing directory.")
_TF_DATA_SERVICE_ADDRESS = flags.DEFINE_string(
    "tf_data_service_address",
    "",
    "The dispatcher's address.",
    allow_override_cpp=True)
flags.mark_flags_as_required(["config", "workdir", "logdir"])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  logging.info(_CONFIG.value)
  logging.info(_METHOD_CONFIG.value)
  tf.config.experimental.set_visible_devices([], "GPU")
  config = config_utils.parse_config(_CONFIG.value,
                                     config_globals.get_globals())
  method_config = config_utils.parse_config(_METHOD_CONFIG.value,
                                            config_globals.get_globals())

  # Creating the SDFA method
  da_method = method_config.da_method

  # Grab the source data info we want
  _, source_dataset_info = pipeline.get_dataset(
      tf_data_service_address=_TF_DATA_SERVICE_ADDRESS.value,
      **config.train_data_config)

  # Grab the data used for adaptation
  adaptation_dataset, adaptation_dataset_info = eval_dataset, eval_dataset_info = pipeline.get_dataset(
      tf_data_service_address=_TF_DATA_SERVICE_ADDRESS.value,
      **config.adaptation_data_config)

  if adaptation_dataset_info.features[
      "audio"].sample_rate != config.sample_rate_hz:
    raise ValueError(
        "Dataset sample rate must match config sample rate. To address this, "
        "need to set the sample rate in the config to {}.".format(
            adaptation_dataset_info.features["audio"].sample_rate))

  # Grab the data used for evaluation
  eval_dataset, eval_dataset_info = pipeline.get_dataset(
      tf_data_service_address=_TF_DATA_SERVICE_ADDRESS.value,
      **config.eval_data_config)

  if eval_dataset_info.features["audio"].sample_rate != config.sample_rate_hz:
    raise ValueError(
        "Dataset sample rate must match config sample rate. To address this, "
        "need to set the sample rate in the config to {}.".format(
            eval_dataset_info.features["audio"].sample_rate))

  # Initialize state and bundles
  model_bundles, adaptation_state, key = da_method.initialize(
      source_dataset_info, config.init_config.model_config,
      config.init_config.rng_seed, config.init_config.input_size,
      config.init_config.pretrained_ckpt_dir, **method_config)

  # Perform adaptation
  adaptation_state = adapt.perform_adaptation(
      key=key,
      da_method=da_method,
      adaptation_state=adaptation_state,
      adaptation_dataset=adaptation_dataset,
      model_bundles=model_bundles,
      logdir=_LOGDIR.value,
      num_epochs=method_config.num_epochs)

  # Evaluate
  adapt.evaluate(
      model_bundles=model_bundles,
      adaptation_state=adaptation_state,
      eval_dataset=eval_dataset,
      logdir=_LOGDIR.value,
  )


if __name__ == "__main__":
  app.run(main)
