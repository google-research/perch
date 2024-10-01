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

"""Audio loader function helpers."""

import os
from typing import Callable

from chirp import audio_utils
from etils import epath
import numpy as np


def make_filepath_loader(
    audio_globs: dict[str, tuple[str, str]],
    sample_rate_hz: int = 32000,
    window_size_s: float = 5.0,
    dtype: str = 'float32',
) -> Callable[[str, float], np.ndarray]:
  """Create a function for loading audio from a source ID and offset.

  Note that if multiple globs match a given source ID, the first match is used.

  Args:
    audio_globs: Mapping from dataset name to pairs of `(root directory, file
      glob)`. (See `embed.EmbedConfig` for details.)
    sample_rate_hz: Sample rate of the audio.
    window_size_s: Window size of the audio.
    dtype: Data type of the audio.

  Returns:
    Function for loading audio from a source ID and offset.

  Raises:
    ValueError if no audio path is found for the given source ID.
  """

  def loader(source_id: str, offset_s: float) -> np.ndarray:
    found_path = None
    for base_path, _ in audio_globs.values():
      path = epath.Path(base_path) / source_id
      if path.exists():
        found_path = path
        break
    if found_path is None:
      raise ValueError('No audio path found for source_id: ', source_id)
    return np.array(
        audio_utils.load_audio_window(
            found_path.as_posix(),
            offset_s,
            sample_rate=sample_rate_hz,
            window_size_s=window_size_s,
        ),
        dtype=dtype,
    )

  return loader
