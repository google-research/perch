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

"""TensorFlow Datasets extensions for common acoustic archive formats."""

import abc
import dataclasses
import logging
from typing import Any, Iterable

from chirp import audio_utils
from chirp.data import tfds_features
from etils import epath
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass
class WavDirectoryBuilderConfig(tfds.core.BuilderConfig):
  sample_rate_hz: int = 16_000
  interval_length_s: float = 6.0
  max_peaks: int | None = None

  @property
  def context_duration_samples(self) -> int:
    return int(round(self.interval_length_s * self.sample_rate_hz))


def _generate_context_windows(
    wav_path: epath.Path, config: WavDirectoryBuilderConfig
) -> Iterable[tuple[str, dict[str, Any]]]:
  """Generates audio context window feature dicts from a single mono WAV file.

  Args:
    wav_path: A path readable by scipy.io.wavfile.
    config: Desired properties of the extracted context windows.

  Yields:
    Key-values, where the key is {filename}:{start_time}ms and the value is a
    feature dict conforming to WavDirectoryBuilder._info.
  """
  wavfile = tfds.core.lazy_imports.scipy.io.wavfile

  filename = str(wav_path)

  try:
    with wav_path.open('rb') as f:
      sample_rate, samples = wavfile.read(f)
  except ValueError as e:
    # One case: a file with name ending in .wav starts with several 0 bytes.
    logging.warning('skipped %s due to read() error: %s', wav_path, e)
    return
  assert len(samples.shape) == 1  # following code assumes mono
  samples = samples.astype(np.float32) / -np.iinfo(np.int16).min

  context_duration = config.context_duration_samples
  segment_starts = set()
  max_peaks = config.max_peaks
  if max_peaks:
    peak_indices = audio_utils.find_peaks_from_audio(
        samples, sample_rate, max_peaks
    )
    peak_indices = np.asarray(peak_indices)
    for midpoint in peak_indices:
      segment_start = max(0, midpoint - context_duration // 2)
      segment_starts.add(segment_start)
  else:
    segment_starts.update(range(0, len(samples), context_duration))

  # Assertion failures saying "two examples share the same hashed key" have
  # been observed from full-scale data. Here we'll guard against that by
  # explicitly ensuring no duplicate keys are emitted from a single file.
  keys_emitted = set()

  for segment_start in sorted(segment_starts):
    segment_end = segment_start + context_duration
    if segment_end > len(samples):
      break
    context_window = samples[segment_start:segment_end]
    start_ms = int(round(segment_start / config.sample_rate_hz * 1000))

    key = f'{wav_path}:{start_ms:010d}ms'
    if key in keys_emitted:
      logging.warning('skipped yielding features for duplicate key: %s', key)
      continue
    yield key, {
        'audio': context_window,
        'segment_start': segment_start,
        'segment_end': segment_end,
        'filename': filename,
    }
    keys_emitted.add(key)


class WavDirectoryBuilder(tfds.core.GeneratorBasedBuilder):
  """Abstract base class for reading a nested directory of mono WAV files.

  This provides the WAV reading, slicing into context windows, and a
  configuration that filters to only windows with peaks. Concrete subclasses
  need should set VERSION and RELEASE_NOTES and implement _description and
  _citation.
  """

  BUILDER_CONFIGS = [
      # pylint: disable=unexpected-keyword-arg
      WavDirectoryBuilderConfig(
          name='unfiltered',
          description=(
              'Context windows covering the entire dataset with no overlap.'
          ),
      ),
      WavDirectoryBuilderConfig(
          name='slice_peaked',
          description=(
              'Context windows filtered to five peaks per original file.'
          ),
          max_peaks=5,
      )
      # pylint: enable=unexpected-keyword-arg
  ]

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
      Copy, into tensorflow_datasets/downloads/manual, a nested directory
      structure containing the .wav files to be ingested.
  """

  @abc.abstractmethod
  def _description(self) -> str:
    raise NotImplementedError()

  @abc.abstractmethod
  def _citation(self) -> str:
    raise NotImplementedError()

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=self._description(),
        features=tfds.features.FeaturesDict({
            'audio': tfds_features.Int16AsFloatTensor(
                shape=[self.builder_config.context_duration_samples],
                sample_rate=self.builder_config.sample_rate_hz,
                encoding=tfds.features.Encoding.ZLIB,
            ),
            'segment_start': tfds.features.Scalar(dtype=tf.uint64),
            'segment_end': tfds.features.Scalar(dtype=tf.uint64),
            'filename': tfds.features.Text(),
        }),
        supervised_keys=None,
        citation=self._citation(),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {
        'train': self._generate_examples(dl_manager.manual_dir),
    }

  def _generate_examples(self, root_dir: epath.PathLike):
    """Walks a directory and generates fixed duration slices of all WAV files.

    Args:
      root_dir: Directory path from which to read WAV files.

    Returns:
      PTransform from a WAV file path to a generator key-value pairs
        [filename:start_millis, Example dict].
    """
    beam = tfds.core.lazy_imports.apache_beam

    wav_paths = []

    def _walk(wav_dir: epath.Path):
      """Manually walks the tree under root_dir, collecting WAV paths."""
      # needed because epath intentionally does not implement recursive glob.
      for entry in wav_dir.iterdir():
        if entry.is_file() and (entry.suffix in ['.wav', '.WAV']):
          wav_paths.append(entry)
        if entry.is_dir():
          _walk(entry)

    _walk(epath.Path(root_dir))

    return beam.Create(wav_paths) | beam.ParDo(
        _generate_context_windows, config=self.builder_config
    )
