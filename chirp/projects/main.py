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

r"""Entry point for project scripts.

This binary provides a common entry point for all project scripts. In order to
be compatible, a project must provide a `run` callable which accepts the
arguments `mode` (e.g., `train`, `eval`, `finetune`), a `config` in the form of
a `ConfigDict`, and a `workdir` where temporary files can be stored. Finally,
the `tf_data_service_address` argument is a string which is empty or contains
the address of the tf.data service dispatcher.

If the target does not use TensorFlow at all (i.e., no TFDS or `tf.data`) then
pass `--notf` to avoid importing TensorFlow.
"""

from typing import Protocol, Sequence

from absl import app
from absl import flags
from absl import logging
from chirp import config_utils
from chirp.configs import config_globals
from chirp.train import classifier
from chirp.train import hubert
from chirp.train import mae
from chirp.train import separator
from ml_collections import config_dict
from ml_collections.config_flags import config_flags

from xmanager import xm  # pylint: disable=unused-import


class Run(Protocol):
  """Protocol for entry points of project scripts.

  These scripts should aim to include project-specific arguments into the config
  argument as much as possible, since updating this interface would require
  changing every project that uses this entry point.
  """

  def __call__(
      self,
      mode: str,
      config: config_dict.ConfigDict,
      workdir: str,
      tf_data_service_address: str,
  ):
    ...


TARGETS: dict[str, Run] = {
    "classifier": classifier.run,
    "mae": mae.run,
    "hubert": hubert.run,
    "separator": separator.run,
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
_TF = flags.DEFINE_bool("tf", True, "Whether or not the script uses TF.")
flags.mark_flags_as_required(["config", "workdir", "target", "mode"])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  logging.info(_CONFIG.value)
  if _TF.value:
    # We assume that scripts use JAX, so here we prevent TensorFlow from
    # reserving all the GPU memory (which leaves nothing for JAX to use).
    import tensorflow as tf  # pylint: disable=g-import-not-at-top

    tf.config.experimental.set_visible_devices([], "GPU")
  config = config_utils.parse_config(
      _CONFIG.value, config_globals.get_globals()
  )

  TARGETS[_TARGET.value](
      _MODE.value,
      config,
      _WORKDIR.value,
      _TF_DATA_SERVICE_ADDRESS.value,
  )


if __name__ == "__main__":
  app.run(main)
