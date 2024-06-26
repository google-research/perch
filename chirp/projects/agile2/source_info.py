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
from etils import epath
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
class AudioSources:
  """Mapping from dataset name to root directory and file glob."""

  audio_globs: dict[str, tuple[str, str]]

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
      shard_len_s: float = -1,
      max_shards_per_file: int = -1,
      drop_short_shards: bool = False,
  ) -> Iterator[SourceId]:
    """Yields all sources for all datasets.

    Args:
      shard_len_s: Length of each audio shard. If less than zero, yields one
        SourceId per file.
      max_shards_per_file: Maximum number of shards to yield per file. If less
        than zero, yields all shards.
      drop_short_shards: If True, drop shards that are shorter than the
        shard_len_s (ie, the final shard).
    """
    for dataset_name, (root_dir, file_glob) in self.audio_globs.items():
      filepaths = tuple(epath.Path(root_dir).glob(file_glob))
      for filepath in tqdm.tqdm(filepaths):
        file_id = filepath.as_posix()[len(root_dir) + 1 :]
        if shard_len_s < 0:
          yield SourceId(
              dataset_name=dataset_name,
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
        while shard_num < max_shards_per_file or max_shards_per_file <= 0:
          offset_s = shard_num * shard_len_s
          if offset_s >= audio_len_s:
            break
          if offset_s + shard_len_s > audio_len_s and drop_short_shards:
            break
          yield SourceId(
              dataset_name=dataset_name,
              file_id=file_id,
              offset_s=offset_s,
              shard_len_s=shard_len_s,
              filepath=filepath.as_posix(),
          )
          shard_num += 1
