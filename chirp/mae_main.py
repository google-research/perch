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

"""Train a masked autoencoder."""
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from chirp import config_utils
from chirp import mae_train as train
from chirp import train as finetune
from chirp.configs import config_globals
from chirp.data import pipeline
from ml_collections.config_flags import config_flags
import tensorflow as tf

from xmanager import xm  # pylint: disable=unused-import

TRAIN = "train"
FINETUNE = "finetune"
EVAL = "eval"

_CONFIG = config_flags.DEFINE_config_file("config")
_LOGDIR = flags.DEFINE_string("logdir", None, "Work unit logging directory.")
_WORKDIR = flags.DEFINE_string("workdir", None,
                               "Work unit checkpointing directory.")
_MODE = flags.DEFINE_enum("mode", TRAIN, [TRAIN, FINETUNE, EVAL], "Mode.")
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
  tf.config.experimental.set_visible_devices([], "GPU")
  config = config_utils.parse_config(_CONFIG.value,
                                     config_globals.get_globals())

  if _MODE.value in (TRAIN, FINETUNE):
    train_dataset, dataset_info = pipeline.get_dataset(
        is_train=True,
        tf_data_service_address=_TF_DATA_SERVICE_ADDRESS.value,
        **config.train_dataset_config)
  elif _MODE.value == EVAL:
    valid_dataset, dataset_info = pipeline.get_dataset(
        **config.eval_dataset_config)
  if dataset_info.features["audio"].sample_rate != config.sample_rate_hz:
    raise ValueError(
        "Dataset sample rate must match config sample rate. To address this, "
        "need to set the sample rate in the config to {}.".format(
            dataset_info.features["audio"].sample_rate))

  if _MODE.value == TRAIN:
    model_bundle, train_state = train.initialize_model(
        workdir=_WORKDIR.value, **config.init_config)
  else:
    model_bundle, train_state = train.initialize_finetune_model(
        workdir=_WORKDIR.value, **config.init_config)
  if _MODE.value == TRAIN:
    train_state = model_bundle.ckpt.restore_or_initialize(train_state)
    train.train(
        model_bundle,
        train_state,
        train_dataset,
        logdir=_LOGDIR.value,
        **config.train_config)
  if _MODE.value == FINETUNE:
    train_state = model_bundle.ckpt.restore_or_initialize(train_state)
    finetune.train(
        model_bundle,
        train_state,
        train_dataset,
        logdir=_LOGDIR.value,
        **config.train_config)
  elif _MODE.value == EVAL:
    finetune.evaluate_loop(
        model_bundle,
        train_state,
        valid_dataset,
        workdir=_WORKDIR.value,
        logdir=_LOGDIR.value,
        **config.eval_config)


if __name__ == "__main__":
  app.run(main)
