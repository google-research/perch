# coding=utf-8
# Copyright 2023 The Chirp Authors.
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
from chirp.configs import config_globals
from chirp.train import classifier
from chirp.train import hubert
from chirp.train import mae
from chirp.train import separator
from ml_collections.config_flags import config_flags
import tensorflow as tf

from xmanager import xm  # pylint: disable=unused-import

TARGETS = {
    "classifier": classifier,
    "mae": mae,
    "hubert": hubert,
    "separator": separator,
}

_CONFIG = config_flags.DEFINE_config_file("config")
_WORKDIR = flags.DEFINE_string(
    "workdir", None, "Work unit checkpointing directory."
)
_TARGET = flags.DEFINE_enum(
    "target", None, TARGETS.keys(), "The module to run."
)
_MODE = flags.DEFINE_string("mode", None, "The mode to run.")
_TF_DATA_SERVICE_ADDRESS = flags.DEFINE_string(
    "tf_data_service_address",
    "",
    "The dispatcher's address.",
    allow_override_cpp=True,
)
flags.mark_flags_as_required(["config", "workdir", "target", "mode"])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  logging.info(_CONFIG.value)
  tf.config.experimental.set_visible_devices([], "GPU")
  config = config_utils.parse_config(
      _CONFIG.value, config_globals.get_globals()
  )

  TARGETS[_TARGET.value].run(
      _MODE.value,
      config,
      _WORKDIR.value,
      _TF_DATA_SERVICE_ADDRESS.value,
  )


if __name__ == "__main__":
  app.run(main)
