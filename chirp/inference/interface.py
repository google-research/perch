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

"""Interface for models producing embeddings."""

import dataclasses
import json
from typing import Any, Callable, Dict

from absl import logging
from chirp.taxonomy import namespace
from etils import epath
import librosa
from ml_collections import config_dict
import numpy as np
import tensorflow as tf

LogitType = Dict[str, np.ndarray]

NULL_LOGIT = -20.0
POOLING_METHODS = ['first', 'mean', 'max', 'mid', 'flatten', 'squeeze']


@dataclasses.dataclass
class InferenceOutputs:
  """Wrapper class for outputs from an inference model.

  Attributes:
    embeddings: Embeddings array with shape [Frames, Channels, Features].
    logits: Dictionary mapping a class list L's name to an array of logits. The
      logits array has shape [Frames, L.size] or [Frames, Channels, L.size].
    separated_audio: Separated audio channels with shape [Channels, Samples].
    batched: If True, each output has an additonal batch dimension.
    frontend: Model frontend output, such as melspectrogram.
  """

  embeddings: np.ndarray | None = None
  logits: LogitType | None = None
  separated_audio: np.ndarray | None = None
  batched: bool = False
  frontend: np.ndarray | None = None

  def __post_init__(self):
    # In some scenarios, we may be passed TF EagerTensors. We dereference these
    # to numpy arrays for broad compatibility.
    if hasattr(self.embeddings, 'numpy'):
      self.embeddings = self.embeddings.numpy()
    if self.logits is not None:
      for k, v in self.logits.items():
        if hasattr(v, 'numpy'):
          self.logits[k] = v.numpy()
    if hasattr(self.separated_audio, 'numpy'):
      self.separated_audio = self.separated_audio.numpy()

  def pooled_embeddings(
      self, time_pooling: str, channel_pooling: str = ''
  ) -> np.ndarray:
    """Reduce embeddings over the time and/or channel axis."""
    # Shape is either [B, F, C, D] or [F, C, D], so the time axis is -3.
    outputs = pool_axis(self.embeddings, -3, time_pooling)
    outputs = pool_axis(outputs, -2, channel_pooling)
    return outputs


EmbedFnType = Callable[[np.ndarray], InferenceOutputs]


@dataclasses.dataclass
class EmbeddingModel:
  """Wrapper for a model which produces audio embeddings.

  It is encouraged to implement either the `embed` or `batch_embed` function
  and use a convenience method (`batch_embed_from_embed_fn` or
  `embed_from_batch_embed_fn`) to get the other. It is preferable to implement
  `batch_embed` so long as the model accepts batch input, as batch input
  inference can be much faster.

  Attributes:
    sample_rate: Sample rate in hz.
  """

  sample_rate: int

  @classmethod
  def from_config(
      cls, model_config: config_dict.ConfigDict
  ) -> 'EmbeddingModel':
    """Load the model from a configuration dict."""
    raise NotImplementedError

  def embed(self, audio_array: np.ndarray) -> InferenceOutputs:
    """Create InferenceOutputs from an audio array.

    Args:
      audio_array: An array with shape [Time] containing unit-scaled audio.

    Returns:
      An InferenceOutputs object.
    """
    raise NotImplementedError

  def batch_embed(self, audio_batch: np.ndarray) -> InferenceOutputs:
    """Create InferenceOutputs from a batch of audio arrays.

    Args:
      audio_batch: An array with shape [Time] containing unit-scaled audio.

    Returns:
      An InferenceOutputs object.
    """
    raise NotImplementedError

  def frame_audio(
      self,
      audio_array: np.ndarray,
      window_size_s: float | None,
      hop_size_s: float,
  ) -> np.ndarray:
    """Helper function for framing audio for inference along the last axis."""
    if window_size_s is None or window_size_s < 0:
      return np.expand_dims(audio_array, axis=-2)
    frame_length = int(window_size_s * self.sample_rate)
    hop_length = int(hop_size_s * self.sample_rate)
    if audio_array.shape[-1] < frame_length:
      audio_array = librosa.util.pad_center(
          audio_array, size=frame_length, axis=-1
      )
    # Librosa frames as [..., frame_length, frames], so we need a transpose.
    framed_audio = librosa.util.frame(
        audio_array,
        frame_length=frame_length,
        hop_length=hop_length,
        axis=-1,
    ).swapaxes(-1, -2)
    return framed_audio

  def normalize_audio(
      self,
      framed_audio: np.ndarray,
      target_peak: float | None,
  ) -> np.ndarray:
    """Normalizes audio with shape [..., T] to match the target_peak value."""
    if target_peak is None:
      return framed_audio
    framed_audio = framed_audio.copy()
    framed_audio -= np.mean(framed_audio, axis=-1, keepdims=True)
    peak_norm = np.max(np.abs(framed_audio), axis=-1, keepdims=True)
    framed_audio = np.divide(framed_audio, peak_norm, where=(peak_norm > 0.0))
    framed_audio = framed_audio * target_peak
    return framed_audio


@dataclasses.dataclass
class LogitsOutputHead:
  """A TensorFlow model which classifies embeddings.

  Attributes:
    model_path: Path to saved model.
    logits_key: Name of this output head.
    logits_model: Callable model converting embeddings of shape [B,
      embedding_width] to [B, num_classes].
    class_list: ClassList specifying the ordered classes.
    channel_pooling: Pooling to apply to channel dimension of logits. Specify an
      empty string to apply no pooling.
  """

  model_path: str
  logits_key: str
  logits_model: Any
  class_list: namespace.ClassList
  channel_pooling: str = 'max'

  @classmethod
  def from_config_file(cls, model_path: str, filename='logits_config.json'):
    config_filepath = epath.Path(model_path) / filename
    with (config_filepath).open() as f:
      logits_config = config_dict.ConfigDict(json.loads(f.read()))
    logits_config.model_path = model_path
    return cls.from_config(logits_config)

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict):
    logits_model = tf.saved_model.load(config.model_path)
    model_path = epath.Path(config.model_path)
    with (model_path / 'class_list.csv').open('r') as f:
      class_list = namespace.ClassList.from_csv(f)
    return cls(
        logits_model=logits_model,
        class_list=class_list,
        **config,
    )

  def __call__(self, embeddings: np.ndarray) -> InferenceOutputs:
    """Apply the wrapped logits_model to embeddings with shape [B, D]."""
    if callable(self.logits_model):
      logits = self.logits_model(embeddings)
    elif hasattr(self.logits_model, 'signatures'):
      # TODO(tomdenton): Figure out why the Keras saved model isn't callable.
      flat_logits = self.logits_model.signatures['serving_default'](
          inputs=embeddings
      )
      logits = flat_logits['output_0']
      if hasattr(logits, 'numpy'):
        logits = logits.numpy()
    else:
      raise ValueError('could not figure out how to call wrapped model.')
    return logits

  def save_model(self, output_path: str, embeddings_path: str):
    """Write a SavedModel and metadata to disk."""
    # Write the model.
    tf.saved_model.save(self.logits_model, output_path)
    output_path = epath.Path(output_path)
    # Write a config file.
    config_data = dataclasses.asdict(self)
    for k in ['logits_model', 'class_list']:
      # These are loaded automatically.
      config_data.pop(k)
    with (output_path / 'logits_config.json').open('w') as f:
      json.dump(config_data, f)
    # Copy the embeddings_config if provided
    if embeddings_path:
      (epath.Path(embeddings_path) / 'config.json').copy(
          output_path / 'embeddings_config.json', overwrite=True
      )
    # Write the class list.
    with (output_path / 'class_list.csv').open('w') as f:
      f.write(self.class_list.to_csv())

  def add_logits(self, model_outputs: InferenceOutputs, keep_original: bool):
    """Update the model_outputs to include logits from this output head."""
    embeddings = model_outputs.embeddings
    if embeddings is None:
      logging.warning('No embeddings found in model outputs.')
      return model_outputs
    flat_embeddings = np.reshape(embeddings, [-1, embeddings.shape[-1]])
    flat_logits = self(flat_embeddings)
    logits_shape = np.concatenate(
        [np.shape(embeddings)[:-1], np.shape(flat_logits)[-1:]], axis=0
    )
    logits = np.reshape(flat_logits, logits_shape)
    # Embeddings have shape [B, T, C, D] or [T, C, D], so our logits also
    # have a channel dimension.
    # Output logits should have shape [B, T, D] or [T, D], so we reduce the
    # channel axis as specified by the user.
    # The default is 'max' which is reasonable for separated audio and
    # is equivalent to 'squeeze' for the single-channel case.
    logits = pool_axis(logits, -2, self.channel_pooling)
    new_outputs = InferenceOutputs(
        embeddings=model_outputs.embeddings,
        logits=model_outputs.logits,
        separated_audio=model_outputs.separated_audio,
        batched=model_outputs.batched,
    )
    if new_outputs.logits is None or not keep_original:
      new_outputs.logits = {}
    new_outputs.logits[self.logits_key] = logits
    return new_outputs


def embed_from_batch_embed_fn(
    embed_fn: EmbedFnType, audio_array: np.ndarray
) -> InferenceOutputs:
  """Embed a single example using a batch_embed_fn."""
  audio_batch = audio_array[np.newaxis, :]
  outputs = embed_fn(audio_batch)

  unbatched_outputs = {}
  for k in ['embeddings', 'separated_audio', 'frontend']:
    if getattr(outputs, k) is not None:
      unbatched_outputs[k] = getattr(outputs, k)[0]
    else:
      unbatched_outputs[k] = None

  if outputs.logits is not None:
    logits = {}
    for k, v in outputs.logits.items():
      logits[k] = v[0]
  else:
    logits = None

  return InferenceOutputs(
      logits=logits,
      batched=False,
      **unbatched_outputs,
  )


def batch_embed_from_embed_fn(
    embed_fn: EmbedFnType, audio_batch: np.ndarray
) -> InferenceOutputs:
  """Embed a batch of audio using a single-example embed_fn."""
  outputs = []
  for audio in audio_batch:
    outputs.append(embed_fn(audio))

  batched_outputs = {}
  for k in ['embeddings', 'separated_audio', 'frontend']:
    if getattr(outputs[0], k) is not None:
      batched_outputs[k] = np.stack([getattr(x, k) for x in outputs], axis=0)
    else:
      batched_outputs[k] = None

  if outputs[0].logits is not None:
    batched_logits = {}
    for logit_key in outputs[0].logits:
      batched_logits[logit_key] = np.stack(
          [x.logits[logit_key] for x in outputs], axis=0
      )
  else:
    batched_logits = None

  return InferenceOutputs(
      logits=batched_logits,
      batched=True,
      **batched_outputs,
  )


def pool_axis(ar: np.ndarray, axis: int, pooling: str) -> np.ndarray:
  """Apply the specified pooling along the target axis."""
  if pooling == 'first':
    outputs = ar.take(0, axis=axis)
  elif pooling == 'squeeze':
    # Like 'first' but throws an exception if more than one time step.
    outputs = ar.squeeze(axis=axis)
  elif pooling == 'mean':
    outputs = ar.mean(axis=axis)
  elif pooling == 'max':
    outputs = ar.max(axis=axis)
  elif pooling == 'mid':
    midpoint_index = ar.shape[axis] // 2
    outputs = ar.take(midpoint_index, axis=axis)
  elif pooling == 'flatten':
    # Flatten the target axis dimension into the last dimension.
    outputs = ar.swapaxes(axis, -2)
    new_shape = outputs.shape[:-2] + (outputs.shape[-1] * outputs.shape[-2],)
    outputs = outputs.reshape(new_shape)
  elif not pooling:
    outputs = ar
  else:
    raise ValueError(f'Unrecognized pooling method {pooling}.')
  return outputs
