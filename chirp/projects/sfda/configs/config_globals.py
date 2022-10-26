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

"""Define the globals that can be used in configuration files."""
from typing import Any, Dict

from chirp import audio_utils
from chirp.data import pipeline
from chirp.eval import eval_lib
from chirp.models import conformer
from chirp.models import efficientnet
from chirp.models import frontend
from chirp.models import hubert
from chirp.models import layers
from chirp.models import quantizers
from chirp.models import soundstream_unet
from chirp.projects.sfda.data import pipeline as sfda_pipeline
from chirp.projects.sfda.methods import ada_bn
from chirp.projects.sfda.methods import notela
from chirp.projects.sfda.methods import pseudo_label
from chirp.projects.sfda.methods import shot
from chirp.projects.sfda.methods import tent
from flax import linen as nn


def get_globals() -> Dict[str, Any]:
  return {
      "audio_utils": audio_utils,
      "conformer": conformer,
      "efficientnet": efficientnet,
      "eval_lib": eval_lib,
      "hubert": hubert,
      "quantizers": quantizers,
      "frontend": frontend,
      "layers": layers,
      "nn": nn,
      "pipeline": pipeline,
      "sfda_pipeline": sfda_pipeline,
      "soundstream_unet": soundstream_unet,
      "pseudo_label": pseudo_label,
      "tent": tent,
      "notela": notela,
      "shot": shot,
      "ada_bn": ada_bn,
  }
