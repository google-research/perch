# coding=utf-8
# Copyright 2023 The BIRB Authors.
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

"""Define the globals that can be used in configuration files."""
from typing import Any

from birb import audio_utils
from birb import config_utils
from birb.eval import callbacks
from birb.eval import eval_lib
from birb.models import conformer
from birb.models import efficientnet
from birb.models import frontend
from birb.models import handcrafted_features
from birb.models import layers
from birb.models import taxonomy_model
from birb.preprocessing import pipeline
from flax import linen as nn
import optax


def get_globals() -> dict[str, Any]:
  return {
      "audio_utils": audio_utils,
      "callbacks": callbacks,
      "config_utils": config_utils,
      "conformer": conformer,
      "efficientnet": efficientnet,
      "eval_lib": eval_lib,
      "frontend": frontend,
      "layers": layers,
      "nn": nn,
      "optax": optax,
      "pipeline": pipeline,
      "handcrafted_features": handcrafted_features,
      "taxonomy_model": taxonomy_model,
  }
