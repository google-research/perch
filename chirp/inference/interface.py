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

from chirp.taxonomy import namespace
import librosa
import numpy as np

LogitType = Dict[str, np.ndarray]

NULL_LOGIT = -20.0


@dataclasses.dataclass
class InferenceOutputs:
  """Wrapper class for outputs from an inference model.

  Attributes:
    embeddings: Embeddings array with shape [Time, Channels, Features].
    logits: Dictionary mapping a class list L's name to an array of logits. The
      logits array has shape [Time, L.size].
    separated_audio: Separated audio channels with shape [Channels, Samples].
  """
  embeddings: Optional[np.ndarray] = None
  logits: Optional[LogitType] = None
  separated_audio: Optional[np.ndarray] = None


@dataclasses.dataclass
class EmbeddingModel:
  """Wrapper for a model which produces audio embeddings.

  Attributes:
    sample_rate: Sample rate in hz.
  """
  sample_rate: int

  def embed(self, audio_array: np.ndarray) -> InferenceOutputs:
    """Create evenly-spaced embeddings for an audio array.

    Args:
      audio_array: An array with shape [Time] containing unit-scaled audio.

    Returns:
      An InferenceOutputs object.
    """
    raise NotImplementedError

  def batch_embed(self, audio_batch: np.ndarray) -> InferenceOutputs:
    """Embed a batch of audio."""
    outputs = []
    for audio in audio_batch:
      outputs.append(self.embed(audio))
    if outputs[0].embeddings is not None:
      embeddings = np.stack([x.embeddings for x in outputs], axis=0)
    else:
      embeddings = None

    if outputs[0].logits is not None:
      batched_logits = {}
      for logit_key in outputs[0].logits:
        batched_logits[logit_key] = np.stack(
            [x.logits[logit_key] for x in outputs], axis=0)
    else:
      batched_logits = None

    if outputs[0].separated_audio is not None:
      separated_audio = np.stack([x.separated_audio for x in outputs], axis=0)
    else:
      separated_audio = None

    return InferenceOutputs(
        embeddings=embeddings,
        logits=batched_logits,
        separated_audio=separated_audio)

  def convert_logits(
      self, logits: np.ndarray, source_class_list: namespace.ClassList,
      target_class_list: Optional[namespace.ClassList]) -> np.ndarray:
    """Convert model logits to logits for a different class list."""
    if target_class_list is None:
      return logits
    sp_matrix, sp_mask = source_class_list.get_class_map_matrix(
        target_class_list)
    # When we convert from ClassList A (used for training) to ClassList B
    # (for inference output) there may be labels in B which don't appear in A.
    # The `sp_mask` tells us which labels appear in both A and B. We set the
    # logit for the new labels to NULL_LOGIT, which corresponds to a probability
    # very close to zero.
    return logits @ sp_matrix + NULL_LOGIT * (1 - sp_mask)

  def frame_audio(self, audio_array: np.ndarray, window_size_s: Optional[float],
                  hop_size_s: float) -> np.ndarray:
    """Helper function for framing audio for inference."""
    if window_size_s is None or window_size_s < 0:
      return audio_array[np.newaxis, :]
    frame_length = int(window_size_s * self.sample_rate)
    hop_length = int(hop_size_s * self.sample_rate)
    # Librosa frames as [frame_length, batch], so need a transpose.
    framed_audio = librosa.util.frame(audio_array, frame_length, hop_length).T
    return framed_audio
