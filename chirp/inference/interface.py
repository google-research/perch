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

"""Interface for models producing embeddings."""

import dataclasses
from typing import Dict, Optional

import numpy as np

LogitType = Dict[str, np.ndarray]


@dataclasses.dataclass
class InferenceOutputs:
  """Wrapper calss for outputs from an inference model.

  Attributes:
    embeddings: Embeddings array with shape [Batch, Time, Features].
    logits: Dictionary mapping a class list L's name to an array of logits. The
      logits array has shape [Batch, Time, L.size].
    separated_audio: Separated audio channels with shape [Batch, Channels,
      Samples].
  """
  embeddings: np.ndarray
  logits: Optional[LogitType] = None
  separated_audio: Optional[np.ndarray] = None


@dataclasses.dataclass
class EmbeddingModel:
  """Wrapper for a model which produces audio embeddings.

  Attributes:
    sample_rate: Sample rate in hz.
    window_size_s: Allowed window size of the model, or -1 if polymorphic.
  """
  sample_rate: int
  window_size_s: float

  def embed(self, audio_array: np.ndarray) -> InferenceOutputs:
    """Create evenly-spaced embeddings for an audio array.

    Args:
      audio_array: An array of shape [Batch, Time] containing unit-scaled audio.
        We assume that all batch elements are from the same source audio.

    Returns:
      An InferenceOutputs object.
    """
    raise NotImplementedError
