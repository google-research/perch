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
from typing import Optional, Sequence, Tuple, Union

from chirp import signal
from flax import struct
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import scipy
from jax.experimental import jax2tf
from scipy import signal as scipy_signal
import tensorflow as tf


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


@struct.dataclass
class NoScalingConfig:
  pass


@struct.dataclass
class STFTParams:
  """STFT Parameters."""
  frame_step: int
  frame_length: int
  nfft: int
  num_padded_samples: int
  overlap: int


ScalingConfig = Union[LogScalingConfig, PCENScalingConfig, NoScalingConfig]


def get_stft_params(sample_rate_hz: int,
                    frame_rate: int,
                    frame_length_secs: float = 0.08) -> STFTParams:
  """Computes STFT parameters from high-level specifications (hz, seconds)."""
  frame_step = sample_rate_hz // frame_rate
  frame_length = int(sample_rate_hz * frame_length_secs)
  # use math because nfft must be a static argument to signal.stft
  nfft = 2**math.ceil(math.log2(frame_length))
  num_padded_samples = (frame_length - frame_step) // 2
  overlap = frame_length - frame_step
  return STFTParams(frame_step, frame_length, nfft, num_padded_samples, overlap)


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def compute_stft(audio: jnp.ndarray,
                 sample_rate_hz: int,
                 frame_rate: int,
                 frame_length_secs: float = 0.08,
                 use_tf_stft: bool = True) -> jnp.ndarray:
  """Computes the STFT of the audio."""
  stft_params = get_stft_params(sample_rate_hz, frame_rate, frame_length_secs)
  if stft_params.num_padded_samples > 0:
    pad_width = ((0, 0),) * (audio.ndim - 1) + (
        (stft_params.num_padded_samples,) * 2,)
    audio = jnp.pad(audio, pad_width)

  if use_tf_stft:
    # The Jax stft uses a complex convolution which is not supported
    # by TFLite.
    def _tf_stft(x):
      return tf.signal.stft(
          x,
          frame_length=stft_params.frame_length,
          frame_step=stft_params.frame_step,
          fft_length=stft_params.nfft,
          pad_end=False)

    stfts = jax2tf.call_tf(_tf_stft)(audio)
  else:
    _, _, stfts = scipy.signal.stft(
        audio,
        sample_rate_hz,
        nperseg=stft_params.frame_length,
        noverlap=stft_params.overlap,
        nfft=stft_params.nfft,
        return_onesided=True,
        window="hann",
        padded=False,
        boundary=None)
    # Scaling to match the tf.signal.stft output.
    stfts = stft_params.frame_length / 2 * stfts
    stfts = jnp.swapaxes(stfts, -1, -2)
  return stfts


def compute_istft(spectrogram: jnp.ndarray,
                  sample_rate_hz: int,
                  frame_rate: int,
                  frame_length_secs: float = 0.08,
                  use_tf_stft: bool = True) -> jnp.ndarray:
  """Applies the inverse STFT."""
  stft_params = get_stft_params(sample_rate_hz, frame_rate, frame_length_secs)

  if use_tf_stft:
    # The Jax stft uses a complex convolution which is not supported
    # by TFLite.
    def _tf_istft(x):
      return tf.signal.inverse_stft(
          x,
          frame_length=stft_params.frame_length,
          frame_step=stft_params.frame_step,
          fft_length=stft_params.nfft,
          window_fn=tf.signal.inverse_stft_window_fn(stft_params.frame_step))

    waveform = jax2tf.call_tf(_tf_istft)(spectrogram)
  else:
    spectrogram = 2 / stft_params.frame_length * spectrogram
    waveform = jax.scipy.signal.istft(
        spectrogram,
        nperseg=stft_params.frame_length,
        nfft=stft_params.nfft,
        boundary=False,
        input_onesided=True,
        time_axis=-2,
        freq_axis=-1,
        noverlap=stft_params.overlap)[1]

  # Remove center padding added by the forward STFT.
  if stft_params.num_padded_samples > 0:
    pad = stft_params.num_padded_samples
    waveform = waveform[..., pad:-pad]
  return waveform


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4, 7))
def compute_melspec(
    audio: jnp.ndarray,
    sample_rate_hz: int,
    melspec_depth: int,
    melspec_frequency: int,
    frame_length_secs: float = 0.08,
    lower_edge_hz: float = 60.0,
    upper_edge_hz: float = 10_000.0,
    use_tf_stft: bool = False,
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
    upper_edge_hz: the desired top edge of the highest frequency band in the mel
      spectrum.
    use_tf_stft: if true, uses the Tensorflow STFT op.
    scaling_config: Scaling configuration.

  Returns:
    The melspectrogram of the audio, shape `(melspec_frequency, melspec_depth)`.
  """
  scaling_config = scaling_config or NoScalingConfig()

  stfts = compute_stft(audio, sample_rate_hz, melspec_frequency,
                       frame_length_secs, use_tf_stft)
  magnitude_spectrograms = jnp.abs(stfts)

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
  elif isinstance(scaling_config, NoScalingConfig):
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


def apply_mixture_denoising(melspec: jnp.ndarray,
                            threshold: float) -> jnp.ndarray:
  """Denoises the melspectrogram using an estimated Gaussian noise distribution.

  Forms a noise estimate by a) estimating mean+std, b) removing extreme
  values, c) re-estimating mean+std for the noise, and then d) classifying
  values in the spectrogram as 'signal' or 'noise' based on likelihood under
  the revised estimate. We then apply a mask to return the signal values.

  Args:
    melspec: input melspectrogram of rank 2 (time, frequency).
    threshold: z-score theshold for separating signal from noise. On the first
      pass, we use 2 * threshold, and on the second pass we use threshold
      directly.

  Returns:
    The denoised melspectrogram.
  """
  x = melspec
  feature_mean = jnp.mean(x, axis=0, keepdims=True)
  feature_std = jnp.std(x, axis=0, keepdims=True)
  is_noise = (x - feature_mean) < 2 * threshold * feature_std

  noise_counts = jnp.sum(is_noise.astype(x.dtype), axis=0, keepdims=True)
  noise_mean = jnp.sum(x * is_noise, axis=1, keepdims=True) / (noise_counts + 1)
  noise_var = jnp.sum(
      is_noise * jnp.square(x - noise_mean), axis=0, keepdims=True)
  noise_std = jnp.sqrt(noise_var / (noise_counts + 1))

  # Recompute signal/noise separation.
  demeaned = x - noise_mean
  is_signal = demeaned >= threshold * noise_std
  is_signal = is_signal.astype(x.dtype)
  is_noise = 1.0 - is_signal

  signal_part = is_signal * x
  noise_part = is_noise * noise_mean
  reconstructed = signal_part + noise_part - noise_mean
  return reconstructed


def pad_to_length_if_shorter(audio: jnp.ndarray, target_length: int):
  """Wraps the audio sequence if it's shorter than the target length.

  Args:
    audio: input audio sequence of shape [num_samples].
    target_length: target sequence length.

  Returns:
    The audio sequence, padded through wrapping (if it's shorter than the target
    length).
  """
  if audio.shape[0] < target_length:
    missing = target_length - audio.shape[0]
    pad_left = missing // 2
    pad_right = missing - pad_left
    audio = jnp.pad(audio, [[pad_left, pad_right]], mode="wrap")
  return audio


def slice_peaked_audio(
    audio: jnp.ndarray,
    sample_rate_hz: int,
    interval_length_s: float = 6.0,
    max_intervals: int = 5,
    scaling_config: Optional[ScalingConfig] = None) -> Sequence[jnp.ndarray]:
  """Extracts audio intervals from melspec peaks.

  Args:
    audio: input audio sequence of shape [num_samples].
    sample_rate_hz: sample rate of the audio sequence (Hz).
    interval_length_s: length each extracted audio interval.
    max_intervals: upper-bound on the number of audio intervals to extract.
    scaling_config: Scaling configuration for the melspectrogram computation.

  Returns:
    Sequence of extracted audio intervals, each of shape
    [sample_rate_hz * interval_length_s].
  """
  scaling_config = scaling_config or PCENScalingConfig()
  target_length = int(sample_rate_hz * interval_length_s)

  # Wrap audio to the target length if it's shorter than that.
  audio = pad_to_length_if_shorter(audio, target_length)

  peaks = find_peaks_from_audio(audio, sample_rate_hz, max_intervals,
                                scaling_config)
  left_shift = target_length // 2
  right_shift = target_length - left_shift

  # Ensure that the peak locations are such that
  # `audio[peak - left_shift: peak + right_shift]` is a non-truncated slice.
  peaks = jnp.clip(peaks, left_shift, audio.shape[0] - right_shift)
  # As a result, it's possible that some (start, stop) pairs become identical;
  # eliminate duplicates.
  start_stop = jnp.unique(
      jnp.stack([peaks - left_shift, peaks + right_shift], axis=-1), axis=0)

  return [audio[a:b] for (a, b) in start_stop]


def find_peaks_from_audio(audio: jnp.ndarray, sample_rate_hz: int,
                          max_peaks: int,
                          scaling_config: ScalingConfig) -> jnp.ndarray:
  """Construct melspec and find peaks.

  Args:
    audio: input audio sequence of shape [num_samples].
    sample_rate_hz: sample rate of the audio sequence (Hz).
    max_peaks: upper-bound on the number of peaks to return.
    scaling_config: Scaling configuration for the melspectrogram computation.

  Returns:
    Sequence of scalar indices for the peaks found in the audio sequence.
  """
  melspec_rate_hz = 100
  melspec = compute_melspec(
      audio=audio,
      sample_rate_hz=sample_rate_hz,
      melspec_depth=160,
      melspec_frequency=melspec_rate_hz,
      scaling_config=scaling_config)
  melspec = apply_mixture_denoising(melspec, 0.75)

  peaks = find_peaks_from_melspec(melspec, melspec_rate_hz)
  peak_energies = jnp.sum(melspec, axis=1)[peaks]

  t_mel_to_t_au = lambda tm: 1.0 * tm * sample_rate_hz / melspec_rate_hz
  peaks = [t_mel_to_t_au(p) for p in peaks]

  peak_set = sorted(zip(peak_energies, peaks), reverse=True)
  if max_peaks > 0 and len(peaks) > max_peaks:
    peak_set = peak_set[:max_peaks]
  return jnp.asarray([p[1] for p in peak_set], dtype=jnp.int32)


def find_peaks_from_melspec(melspec: jnp.ndarray, stft_fps: int) -> jnp.ndarray:
  """Locate peaks inside signal of summed spectral magnitudes.

  Args:
    melspec: input melspectrogram of rank 2 (time, frequency).
    stft_fps: Number of summed magnitude bins per second. Calculated from the
      original sample of the waveform.

  Returns:
    A list of filtered peak indices.
  """
  summed_spectral_magnitudes = jnp.sum(melspec, axis=1)
  threshold = jnp.mean(summed_spectral_magnitudes) * 1.5
  min_width = int(round(0.5 * stft_fps))
  max_width = int(round(2 * stft_fps))
  width_step_size = int(round((max_width - min_width) / 10))
  peaks = scipy_signal.find_peaks_cwt(
      summed_spectral_magnitudes,
      jnp.arange(min_width, max_width, width_step_size))
  margin_frames = int(round(0.3 * stft_fps))
  start_stop = jnp.clip(
      jnp.stack([peaks - margin_frames, peaks + margin_frames], axis=-1), 0,
      summed_spectral_magnitudes.shape[0])
  peaks = [
      p for p, (a, b) in zip(peaks, start_stop)
      if summed_spectral_magnitudes[a:b].max() >= threshold
  ]
  return jnp.asarray(peaks, dtype=jnp.int32)
