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

"""Audio utilities.

General utilities for processing audio and spectrograms.
"""
import functools
import math
from typing import Optional, Tuple, Union

from chirp import signal
from flax import struct
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import scipy


@struct.dataclass
class LogScalingConfig:
  """Configuration for log-scaling of mel-spectrogram."""
  floor: float = 1e-2
  offset: float = 0.0
  scalar: float = 0.1


@struct.dataclass
class PCENScalingConfig:
  """Configuration for PCEN normalization of mel-spectrogram."""
  smoothing_coef: float = 0.1
  gain: float = 0.5
  bias: float = 2.0
  root: float = 2.0
  eps: float = 1e-6


ScalingConfig = Union[LogScalingConfig, PCENScalingConfig]


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def compute_melspec(
    audio: jnp.ndarray,
    sample_rate_hz: int,
    melspec_depth: int,
    melspec_frequency: int,
    frame_length_secs: float = 0.08,
    lower_edge_hz: float = 60.0,
    upper_edge_hz: float = 10_000.0,
    scaling_config: Optional[ScalingConfig] = None) -> jnp.ndarray:
  """Converts audio to melspectrogram.

  Args:
    audio: input audio, with samples in the last dimension.
    sample_rate_hz: sample rate of the input audio (Hz).
    melspec_depth: number of bands in the mel spectrum.
    melspec_frequency: used to determine the number of samples to step when
      computing the stft (frame_step = sample_rate_hz / melspec_frequency).
    frame_length_secs: the stft window length in seconds.
    lower_edge_hz: lower bound on the frequencies to be included in the mel
      spectrum.
    upper_edge_hz: he desired top edge of the highest frequency band in the mel
      spectrum.
    scaling_config: Scaling configuration.

  Returns:
    The melspectrogram of the audio, shape `(melspec_frequency, melspec_depth)`.
  """
  frame_step = sample_rate_hz // melspec_frequency
  frame_length = int(sample_rate_hz * frame_length_secs)
  # use math because nfft must be a static argument to signal.stft
  nfft = 2**math.ceil(math.log2(frame_length))

  num_padded_samples = (frame_length - frame_step) // 2

  if num_padded_samples > 0:
    pad_width = ((0, 0),) * (audio.ndim - 1) + ((num_padded_samples,) * 2,)
    audio = jnp.pad(audio, pad_width)
  overlap = frame_length - frame_step

  _, _, stfts = scipy.signal.stft(
      audio,
      sample_rate_hz,
      nperseg=frame_length,
      noverlap=overlap,
      nfft=nfft,
      return_onesided=True,
      window="hann",
      padded=False,
      boundary=None)
  magnitude_spectrograms = jnp.abs(stfts)

  # Scaling to match the tf.signal.stft output.
  magnitude_spectrograms = frame_length / 2 * magnitude_spectrograms
  magnitude_spectrograms = jnp.swapaxes(magnitude_spectrograms, -1, -2)
  # An energy spectrogram is the magnitude of the complex-valued STFT.
  # A float32 Tensor of shape [batch_size, ?, num_spectrogram_bins].
  num_spectrogram_bins = magnitude_spectrograms.shape[-1]
  mel_matrix = signal.linear_to_mel_weight_matrix(melspec_depth,
                                                  num_spectrogram_bins,
                                                  sample_rate_hz, lower_edge_hz,
                                                  upper_edge_hz)
  mel_spectrograms = jnp.tensordot(magnitude_spectrograms, mel_matrix, 1)

  if isinstance(scaling_config, LogScalingConfig):
    x = jnp.log(
        jnp.maximum(mel_spectrograms, scaling_config.floor) +
        scaling_config.offset)
    x = scaling_config.scalar * x
  elif isinstance(scaling_config, PCENScalingConfig):
    x, _ = fixed_pcen(mel_spectrograms, scaling_config.smoothing_coef,
                      scaling_config.gain, scaling_config.bias,
                      scaling_config.root, scaling_config.eps)
  elif scaling_config is None:
    x = mel_spectrograms
  else:
    raise ValueError("Unrecognized scaling mode.")

  num_frames = audio.shape[-1] // sample_rate_hz * melspec_frequency
  x = x[..., :num_frames, :]
  return x


def ema(xs: jnp.ndarray,
        gamma: float,
        initial_state: Optional[jnp.ndarray] = None,
        axis: int = 0) -> jnp.ndarray:
  """Computes the exponential moving average along one axis."""
  # Bring target axis to front.
  xs = jnp.swapaxes(xs, 0, axis)
  if initial_state is None:
    initial_state = xs[0]

  def ema_fn(state, x):
    new_state = gamma * x + (1.0 - gamma) * state
    return new_state, new_state

  # NOTE: For small batches this is potentially an expensive and inefficient
  # computation, as it requires a loop over potentially long sequences with
  # minimal computation each step. This could be addressed by partially
  # unrolling the loop or by a truncated EMA using convolutions.
  final_state, ys = lax.scan(ema_fn, init=initial_state, xs=xs)

  ys = jnp.swapaxes(ys, 0, axis)
  return ys, final_state


def fixed_pcen(
    filterbank_energy: jnp.ndarray,
    smoothing_coef: float = 0.05638943879134889,
    gain: float = 0.98,
    bias: float = 2.0,
    root: float = 2.0,
    eps: float = 1e-6,
    state: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Per-Channel Energy Normalization (PCEN) with fixed parameters.

  See https://arxiv.org/abs/1607.05666 for details.

  Args:
    filterbank_energy: A [..., num_frames, num_frequency_bins] array of
      power-domain filterbank energies. If a scalar, we return 0.0 as the
      spectral floor value (for padding purposes).
    smoothing_coef: The coefficient of the IIR smoothing filter (scalar or for
      each bin). Referred to as s in the paper.
    gain: The normalization coefficient (scalar or for each bin). Alpha in the
      paper.
    bias: Constant stabilizer offset for the root compression (scalar or for
      each bin). Delta in the paper.
    root: Root compression coefficient (scalar or for each bin). The reciprocal
      of r in the paper.
    eps: Epsilon floor value to prevent division by zero.
    state: Optional state produced by a previous call to fixed_pcen. Used in
      streaming mode.

  Returns:
    Filterbank energies with PCEN compression applied (type and shape are
    unchanged). Also returns a state tensor to be used in the next call to
    fixed_pcen.
  """
  if filterbank_energy.ndim < 2:
    raise ValueError("Filterbank energy must have rank >= 2.")

  for name, arr, max_rank in (("gain", gain, 1), ("bias", bias, 1),
                              ("root", root, 1), ("smoothing_coef",
                                                  smoothing_coef, 0), ("eps",
                                                                       eps, 0)):
    if jnp.ndim(arr) > max_rank:
      raise ValueError(f"{name} must have rank at most {max_rank}")

  smoothed_energy, filter_state = ema(
      filterbank_energy, smoothing_coef, initial_state=state, axis=-2)
  inv_root = 1. / root
  pcen_output = ((filterbank_energy /
                  (eps + smoothed_energy)**gain + bias)**inv_root -
                 bias**inv_root)

  return pcen_output, filter_state


def random_low_pass_filter(key: jnp.ndarray,
                           melspec: jnp.ndarray,
                           time_axis: int = -2,
                           channel_axis: int = -1,
                           min_slope: float = 2.,
                           max_slope: float = 8.,
                           min_offset: float = 0.,
                           max_offset: float = 5.0) -> jnp.ndarray:
  """Applies a random low-pass rolloff frequency envelope.

  Args:
    key: A random key used to sample a random slope and offset.
    melspec: A (batch) of mel-spectrograms, assumed to have frequencies on the
      last axis.
    time_axis: The axis representing time.
    channel_axis: The axis representing the different frequencies.
    min_slope: The minimum slope of the low-pass filter.
    max_slope: The maximum slope of the low-pass filter.
    min_offset: The minimum offset of the low-pass filter.
    max_offset: The maximum offset of the low-pass filte.r

  Returns:
    The mel-spectrogram with a random low-pass filter applied, same size as the
    input.
  """
  shape = list(melspec.shape)
  shape[time_axis] = shape[channel_axis] = 1

  slope_key, offset_key = random.split(key)
  slope = random.uniform(slope_key, shape, minval=min_slope, maxval=max_slope)
  offset = random.uniform(
      offset_key, shape, minval=min_offset, maxval=max_offset)

  shape = [1] * melspec.ndim
  shape[channel_axis] = melspec.shape[channel_axis]
  xspace = jnp.linspace(0.0, 1.0, melspec.shape[channel_axis])
  xspace = jnp.reshape(xspace, shape)

  envelope = 1 - 0.5 * (jnp.tanh(slope * (xspace - 0.5) - offset) + 1)
  return melspec * envelope
