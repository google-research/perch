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

"""Audio source information handling."""

import dataclasses
from typing import Iterator

from absl import logging
from chirp.projects.hoplite import interface as hoplite_interface
from etils import epath
from ml_collections import config_dict
import soundfile
import tqdm


@dataclasses.dataclass
class SourceId:
  """Source information for pairing audio with embeddings."""

  dataset_name: str
  file_id: str
  offset_s: float
  shard_len_s: float
  filepath: str

  def to_id(self):
    return f'{self.dataset_name}:{self.file_id}:{self.offset_s}'


@dataclasses.dataclass
class AudioSourceConfig(hoplite_interface.EmbeddingMetadata):
  """Configuration for embedding a collection of audio sources.

  Attributes:
    dataset_name: Name of the dataset. (Must be unique for each set of files.)
    base_path: Root directory of the dataset.
    file_glob: Glob pattern for the audio files.
    min_audio_len_s: Minimum audio length to process.
    target_sample_rate_hz: Target sample rate for audio. If -2, use the
      embedding model's declared sample rate. If -1, use the file's native
      sample rate. If > 0, resample to the specified rate.
    shard_len_s: If not None, shard the audio into segments of this length.
    max_shards_per_file: If not None, maximum number of shards per file.
  """

  dataset_name: str
  base_path: str
  file_glob: str
  min_audio_len_s: float = 1.0
  target_sample_rate_hz: int = -2
  shard_len_s: float | None = 60.0
  max_shards_per_file: int | None = None

  def is_compatible(self, other: 'AudioSourceConfig') -> bool:
    """Returns True if other is expected to produce comparable embeddings."""
    return (
        self.dataset_name == other.dataset_name
        and self.target_sample_rate_hz == other.target_sample_rate_hz
        and self.min_audio_len_s == other.min_audio_len_s
    )


@dataclasses.dataclass
class AudioSources(hoplite_interface.EmbeddingMetadata):
  """A collection of AudioSourceConfig, with SourceId iterator."""

  audio_globs: tuple[AudioSourceConfig, ...]

  def __post_init__(self):
    dataset_names = set(
        audio_glob.dataset_name for audio_glob in self.audio_globs
    )
    if len(dataset_names) < len(self.audio_globs):
      raise ValueError('Dataset names must be unique.')

  def to_config_dict(self) -> config_dict.ConfigDict:
    """Convert to a config dict."""
    globs = tuple(g.to_config_dict() for g in self.audio_globs)
    return config_dict.ConfigDict({'audio_globs': globs})

  @classmethod
  def from_config_dict(cls, config: config_dict.ConfigDict) -> 'AudioSources':
    """Create an AudioSources from a config dict."""
    globs = tuple(
        AudioSourceConfig(**audio_glob) for audio_glob in config.audio_globs
    )
    return cls(audio_globs=globs)

  def merge_update(self, other: 'AudioSources') -> 'AudioSources':
    """Update the audio sources with the new sources.

    Args:
      other: The new audio sources.

    Raises:
      ValueError if any audio globs appear in both and are incompatible.
    Returns:
      A new AudioSources object with the merged audio globs. In case of a
      conflict, the values in the 'other' audio glob takes precedence.
    """
    my_globs = {g.dataset_name: g for g in self.audio_globs}
    other_globs = {g.dataset_name: g for g in other.audio_globs}
    for dataset_name, my_glob in my_globs.items():
      if dataset_name not in other_globs:
        other_globs[dataset_name] = my_glob
      elif not other_globs[dataset_name].is_compatible(my_glob):
        raise ValueError(
            f'Audio glob {other_globs[dataset_name]} '
            f'is incompatible with {my_glob}.'
        )
    return AudioSources(tuple(other_globs.values()))

  def get_file_length_s(self, filepath: str) -> float:
    """Returns the length of the audio file in seconds."""
    try:
      sf = soundfile.SoundFile(filepath)
      file_length_s = sf.frames / sf.samplerate
      return file_length_s
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.error('Failed to parse audio file (%s) : %s.', filepath, exc)
    return -1

  def iterate_all_sources(
      self,
      target_dataset_name: str | None = None,
  ) -> Iterator[SourceId]:
    """Yields all sources for all datasets (or just a single dataset).

    Args:
      target_dataset_name: If not None, only yield sources for this dataset.

    Yields:
      SourceId objects.
    """
    for glob in self.audio_globs:
      if (
          target_dataset_name is not None
          and glob.dataset_name != target_dataset_name
      ):
        continue
      # If base_path is a URL, the posix path may not match the original string.
      base_path = epath.Path(glob.base_path)
      filepaths = tuple(base_path.glob(glob.file_glob))
      shard_len_s = glob.shard_len_s
      max_shards_per_file = glob.max_shards_per_file

      for filepath in tqdm.tqdm(filepaths):
        file_id = filepath.as_posix()[len(base_path.as_posix()) + 1 :]
        if shard_len_s is None:
          yield SourceId(
              dataset_name=glob.dataset_name,
              file_id=file_id,
              offset_s=0,
              shard_len_s=-1,
              filepath=filepath.as_posix(),
          )
          continue

        # Otherwise, need to emit sharded SourceId's.
        audio_len_s = self.get_file_length_s(filepath)
        if audio_len_s <= 0:
          continue
        shard_num = 0
        while max_shards_per_file is None or shard_num < max_shards_per_file:
          offset_s = shard_num * shard_len_s
          if offset_s >= audio_len_s:
            break
          # When the new shard extends beyond the end of the audio, and the
          # shard will be shorter than the minimum audio length, we are done.
          if (
              offset_s + shard_len_s > audio_len_s
              and audio_len_s - offset_s < glob.min_audio_len_s
          ):
            break
          yield SourceId(
              dataset_name=glob.dataset_name,
              file_id=file_id,
              offset_s=offset_s,
              shard_len_s=shard_len_s,
              filepath=filepath.as_posix(),
          )
          shard_num += 1
