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

"""Train a taxonomy classifier."""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from chirp import config_utils
from chirp import hubert_train
from chirp.configs import config_globals
from chirp.data import pipeline
from ml_collections.config_flags import config_flags
import tensorflow as tf

from xmanager import xm  # pylint: disable=unused-import

TRAIN = "train"
EVAL = "eval"

_CONFIG = config_flags.DEFINE_config_file("config")
_WORKDIR = flags.DEFINE_string("workdir", None,
                               "Work unit checkpointing directory.")
_MODE = flags.DEFINE_enum("mode", TRAIN, [TRAIN, EVAL], "Mode.")
_TF_DATA_SERVICE_ADDRESS = flags.DEFINE_string(
    "tf_data_service_address",
    "",
    "The dispatcher's address.",
    allow_override_cpp=True)
flags.mark_flags_as_required(["config", "workdir"])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  logging.info(_CONFIG.value)
  tf.config.experimental.set_visible_devices([], "GPU")
  config = config_utils.parse_config(_CONFIG.value,
                                     config_globals.get_globals())

  if _MODE.value == TRAIN:
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

  reload_quantizer = False
  if config.init_config.reload_quantizer_from:
    reload_quantizer = True

  # Adjust the multiplier of the quantizer loss such that the quantizer gets the
  # intended starting learning rate.
  quant_start_lr = config.init_config.quant_start_learning_rate
  start_lr = config.init_config.start_learning_rate
  quant_loss_mult = quant_start_lr / start_lr
  quant_loss_mult *= config.train_config.quant_loss_mult

  # Initialize.
  model = hubert_train.initialize_model(
      workdir=_WORKDIR.value,
      num_train_steps=config.train_config.num_train_steps,
      **config.init_config)
  if _MODE.value == TRAIN:
    hubert_train.train(
        *model,
        train_dataset,
        reload_quantizer=reload_quantizer,
        logdir=_WORKDIR.value,
        num_train_steps=config.train_config.num_train_steps,
        log_every_steps=config.train_config.log_every_steps,
        checkpoint_every_steps=config.train_config.checkpoint_every_steps,
        num_quantizer_pretrain_steps=config.train_config
        .num_quantizer_pretrain_steps,
        quant_loss_mult=quant_loss_mult,
        readout_loss_mult=config.train_config.readout_loss_mult,
        hubert_loss_mult=config.train_config.hubert_loss_mult)

  elif _MODE.value == EVAL:
    hubert_train.evaluate_loop(
        *model,
        valid_dataset,
        workdir=_WORKDIR.value,
        logdir=_WORKDIR.value,
        **config.eval_config)


if __name__ == "__main__":
  app.run(main)
