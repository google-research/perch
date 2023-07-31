# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""Create embeddings for an audio corpus."""

import dataclasses
import importlib
import json
import os
from typing import Any, Sequence

from absl import logging
import apache_beam as beam
import audioread
from chirp import audio_utils
from chirp import path_utils
from chirp.inference import interface
from chirp.inference import models
from chirp.inference import tf_examples
from etils import epath
import librosa
from ml_collections import config_dict
import numpy as np
import soundfile
import tensorflow as tf


INFERENCE_CONFIGS_PKG = 'chirp.inference.configs.'


@dataclasses.dataclass
class SourceInfo:
  """Source information for extracting target audio from a file."""

  filepath: str
  shard_num: int
  shard_len_s: float

  def file_id(self, file_id_depth: int) -> str:
    file_id = epath.Path(
        *epath.Path(self.filepath).parts[-(file_id_depth + 1) :]
    ).as_posix()
    return file_id


def create_source_infos(
    source_file_patterns: Sequence[str],
    num_shards_per_file: int,
    shard_len_s: float,
) -> Sequence[SourceInfo]:
  """Expand source file patterns into a list of SourceInfos."""
  # TODO(tomdenton): probe each file and create work units in a new Beam stage.
  source_files = []
  for pattern in source_file_patterns:
    for source_file in epath.Path('').glob(pattern):
      source_files.append(source_file)

  source_file_splits = []
  for source in source_files:
    for i in range(num_shards_per_file):
      source_file_splits.append(SourceInfo(source.as_posix(), i, shard_len_s))
  return source_file_splits


class EmbedFn(beam.DoFn):
  """Beam worker function for creating audio embeddings.

  TODO(tomdenton): Move most of this functionality into the EmbeddingModel.
  This will increase usability in non-beam contexts.
  """

  def __init__(
      self,
      write_embeddings: bool,
      write_logits: bool,
      write_separated_audio: bool,
      write_raw_audio: bool,
      model_key: str,
      model_config: config_dict.ConfigDict,
      crop_s: float = -1.0,
      file_id_depth: int = 0,
      min_audio_s: float = 5.0,
      embedding_model: interface.EmbeddingModel | None = None,
      speech_filter_threshold: float = -1.0,
      speech_filter_width: int = 5,
      target_sample_rate: int = -2,
  ):
    """Initialize the embedding DoFn.

    Args:
      write_embeddings: Whether to write embeddings.
      write_logits: Whether to write output logits.
      write_separated_audio: Whether to write out separated audio tracks.
      write_raw_audio: If true, will add the original audio to the output.
      model_key: String indicating which model wrapper to use. See MODEL_KEYS.
        Only used for setting up the embedding model.
      model_config: Keyword arg dictionary for the model wrapper class. Only
        used for setting up the embedding model.
      crop_s: If greater than zero, run on only the first crop_s seconds.
      file_id_depth: Number of parent directories to include in the file_id. eg,
        If file_id_depth=2 and the filename is `C://my/file/is/awesome.wav`,
        then the file_id will be `file/is/awesome.wav`.
      min_audio_s: Minimum allowed audio length, in seconds.
      embedding_model: Pre-loaded embedding model.
      speech_filter_threshold: Filter threshold for yamnet voice activation. Any
        work unit which contians audio exceeding this threshold will be
        discarded.
      speech_filter_width: Number of timesteps to average when computing the
        speech score.
      target_sample_rate: Target sample rate when loading audio. Set to -2 to
        use the embedding model's native sample rate, or any positive number to
        resample to a fixed rate.
    """
    self.model_key = model_key
    self.model_config = model_config
    self.write_embeddings = write_embeddings
    self.write_logits = write_logits
    self.write_separated_audio = write_separated_audio
    self.write_raw_audio = write_raw_audio
    self.crop_s = crop_s
    self.embedding_model = embedding_model
    self.file_id_depth = file_id_depth
    self.min_audio_s = min_audio_s
    self.speech_filter_threshold = speech_filter_threshold
    self.speech_filter_width = speech_filter_width
    self.target_sample_rate = target_sample_rate

  def setup(self):
    if self.embedding_model is None:
      self.embedding_model = models.model_class_map()[self.model_key](
          **self.model_config
      )
    if hasattr(self, 'model_key'):
      del self.model_key
    if hasattr(self, 'model_config'):
      del self.model_config
    if self.speech_filter_threshold > 0.0:
      self.yamnet = models.TFHubModel.yamnet()
    if self.target_sample_rate == -2:
      self.target_sample_rate = self.embedding_model.sample_rate
    elif self.target_sample_rate > 0:
      self.target_sample_rate = self.target_sample_rate
    else:
      raise ValueError('Invalid target_sample_rate.')

  def load_audio(
      self, filepath: str, offset_s: float, window_size_s: float
  ) -> np.ndarray | None:
    audio = audio_utils.load_audio_window(
        filepath, offset_s, self.target_sample_rate, window_size_s
    )
    logging.warning('Audio loaded successfully.')
    # Convert audio from jax array to numpy array.
    return np.array(audio)

  def get_speech_score(self, audio: np.ndarray) -> float:
    """Check whether the audio contains human speech."""
    if self.speech_filter_threshold <= 0.0:
      return -1.0
    # resample audio to yamnet 16kHz target.
    audio = librosa.resample(
        audio,
        self.target_sample_rate,
        self.yamnet.sample_rate,
        res_type='polyphase',
    )
    speech_logits = self.yamnet.embed(audio).logits['label'][..., 0]

    # Apply a low-pass filter over the yamnet speech logits.
    # This ensures that transient false positives don't ruin our day.
    width = self.speech_filter_width
    speech_logits = (
        np.convolve(speech_logits, np.ones([width]), 'valid') / width
    )
    return speech_logits.max()

  def _log_exception(self, source_info, exception, counter_name):
    beam.metrics.Metrics.counter('beaminference', counter_name).inc()
    logging.warning(
        'The audio at (%s / %d) could not be loaded (%s). '
        'The exception was (%s)',
        source_info.filepath,
        source_info.shard_num,
        counter_name,
        exception,
    )

  def audio_to_example(
      self, file_id: str, timestamp_offset_s: float, audio: np.ndarray
  ) -> tf.train.Example:
    """Embed audio and create a TFExample."""
    if self.embedding_model is None:
      raise ValueError('Embedding model undefined.')
    model_outputs = self.embedding_model.embed(audio)
    example = tf_examples.model_outputs_to_tf_example(
        model_outputs=model_outputs,
        file_id=file_id,
        audio=audio,
        timestamp_offset_s=timestamp_offset_s,
        write_raw_audio=self.write_raw_audio,
        write_separated_audio=self.write_separated_audio,
        write_embeddings=self.write_embeddings,
        write_logits=self.write_logits,
    )
    return example

  @beam.typehints.with_output_types(Any)
  def process(self, source_info: SourceInfo, crop_s: float = -1.0):
    """Process a source.

    Args:
      source_info: SourceInfo describing the audio to process.
      crop_s: If >0, only the first crop_s seconds will be used. Helpful for
        dry-run testing.

    Returns:
      A TFExample.
    """
    file_id = source_info.file_id(self.file_id_depth)

    logging.info('...loading audio (%s)', source_info.filepath)
    timestamp_offset_s = source_info.shard_num * source_info.shard_len_s

    if crop_s > 0:
      window_size_s = crop_s
    elif self.crop_s > 0:
      window_size_s = self.crop_s
    elif source_info.shard_len_s > 0:
      window_size_s = source_info.shard_len_s
    else:
      window_size_s = -1

    try:
      audio = self.load_audio(
          source_info.filepath, timestamp_offset_s, window_size_s
      )
    except soundfile.LibsndfileError as inst:
      self._log_exception(source_info, inst, 'audio_libsndfile_error')
      return
    except ValueError as inst:
      self._log_exception(source_info, inst, 'audio_bad_offset')
      return
    except audioread.NoBackendError as inst:
      self._log_exception(source_info, inst, 'audio_no_backend')
      return
    except EOFError as inst:
      self._log_exception(source_info, inst, 'audio_eof_error')
      return
    except RuntimeError as inst:
      if 'Soundfile is not available' in str(inst):
        self._log_exception(source_info, inst, 'audio_no_soundfile')
      else:
        self._log_exception(source_info, inst, 'audio_runtime_error')
      return

    if audio is None:
      self._log_exception(source_info, 'no_exception', 'audio_empty')
      return
    if audio.shape[0] < self.min_audio_s * self.target_sample_rate:
      self._log_exception(source_info, 'no_exception', 'audio_too_short')
      return
    speech_score = self.get_speech_score(audio)
    if 0 < self.speech_filter_threshold < speech_score:
      self._log_exception(source_info, 'no_exception', 'contains_speech')
      return

    logging.info(
        '...creating embeddings (%s / %d)', file_id, timestamp_offset_s
    )
    example = self.audio_to_example(file_id, timestamp_offset_s, audio)
    beam.metrics.Metrics.counter('beaminference', 'examples_processed').inc()
    return [example]


def get_config(config_key: str):
  """Get a config given its keyed name."""
  module_key = '..{}'.format(config_key)
  config = importlib.import_module(
      module_key, INFERENCE_CONFIGS_PKG
  ).get_config()

  logging.info('Loaded config %s', config_key)
  logging.info('Config output location : %s', config.output_dir)
  return config


def maybe_write_config(parsed_config, output_dir):
  config_json = parsed_config.to_json(indent=2)
  if (output_dir / 'config.json').exists():
    with (output_dir / 'config.json').open('r') as f:
      got_json = f.read()
    if config_json == got_json:
      return
  with (output_dir / 'config.json').open('w') as f:
    f.write(config_json)


def load_embedding_config(embeddings_path):
  """Loads the configuration to generate unlabeled embeddings."""
  embeddings_path = epath.Path(embeddings_path)
  with (embeddings_path / 'config.json').open() as f:
    embedding_config = config_dict.ConfigDict(json.loads(f.read()))
  return embedding_config


def build_run_pipeline(base_pipeline, output_dir, source_infos, embed_fn):
  """Create and run a beam pipeline."""
  _ = (
      base_pipeline
      | beam.Create(source_infos)
      | beam.ParDo(embed_fn)
      # When a file is corrupted and can't be loaded EmbedFn
      # returns None. In this case the lambda below returns false, which then
      # filters it out.
      | beam.Filter(lambda x: x)
      | beam.Reshuffle()
      | beam.io.tfrecordio.WriteToTFRecord(
          os.path.join(output_dir, 'embeddings'),
          coder=beam.coders.ProtoCoder(tf.train.Example),
      )
  )
  metrics = base_pipeline.run().metrics()
  return metrics
