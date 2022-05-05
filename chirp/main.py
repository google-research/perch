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

from chirp import audio_utils
from chirp import train
from chirp.models import efficientnet
from ml_collections import config_dict
from ml_collections.config_flags import config_flags
import tensorflow as tf

from xmanager import xm  # pylint: disable=unused-import

_CONFIG = config_flags.DEFINE_config_file("config")
_LOGDIR = flags.DEFINE_string("logdir", None, "Work unit logging directory.")
_WORKDIR = flags.DEFINE_string("workdir", None,
                               "Work unit checkpointing directory.")
flags.mark_flags_as_required(["config", "workdir", "logdir"])


def parse_config(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Parse the model configuration.

  This converts string-based configuration into the necessary objects.

  Args:
    config: The model configuration. Will be modified in-place.

  Returns:
    The modified model configuration which can be passed to the model
    constructor.
  """
  # Handle model config
  model_config = config.model_config
  with model_config.unlocked():
    if model_config.encoder_.startswith("efficientnet-"):
      model = efficientnet.EfficientNetModel(model_config.encoder_[-2:])
      model_config.encoder = efficientnet.EfficientNet(model)
    else:
      raise ValueError("unknown encoder")
    del model_config.encoder_

  # Handle melspec config
  melspec_config = model_config.melspec_config
  with melspec_config.unlocked():
    # TODO(bartvm): Add scaling config for hyperparameter search
    if melspec_config.scaling == "pcen":
      melspec_config.scaling_config = audio_utils.PCENScalingConfig()
    elif melspec_config.scaling == "log":
      melspec_config.scaling_config = audio_utils.LogScalingConfig()
    elif melspec_config.scaling == "raw":
      melspec_config.scaling_config = None
    del melspec_config.scaling
  return config


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  logging.info(_CONFIG.value)
  tf.config.experimental.set_visible_devices([], "GPU")
  config = parse_config(_CONFIG.value)
  train.train_and_evaluate(
      **config, workdir=_WORKDIR.value, logdir=_LOGDIR.value)


if __name__ == "__main__":
  app.run(main)
