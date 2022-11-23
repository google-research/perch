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

"""Initialializing the registry of image models."""
import enum

from chirp.projects.sfda.models.constant_model import ConstantEncoderModel
from chirp.projects.sfda.models.nrc_resnet import NRCResNet101
from chirp.projects.sfda.models.resnet import ResNet50
from chirp.projects.sfda.models.wideresnet import WideResNet2810


class ImageModelName(enum.Enum):
  """Supported model architectures for image experiments."""
  RESNET = "resnet"
  NRC_RESNET = "nrc_resnet"
  WIDERESNET = "wideresnet"
  CONSTANT = "constant"


MODEL_REGISTRY = {
    ImageModelName.RESNET: ResNet50,
    ImageModelName.NRC_RESNET: NRCResNet101,
    ImageModelName.WIDERESNET: WideResNet2810,
    ImageModelName.CONSTANT: ConstantEncoderModel,
}
