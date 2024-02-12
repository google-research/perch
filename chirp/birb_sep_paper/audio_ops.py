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

"""Basic reuable audio transformations. Like melspec."""

from chirp.birb_sep_paper import pcen_ops
import numpy as np
from scipy import signal
import tensorflow as tf

MELSPEC_PARAMS_BASE = {
    'frame_length_secs': 0.08,
    'lower_edge_hertz': 60.0,
    'upper_edge_hertz': 10000.0,
}


def Melspec(
    audio,
    sample_frequency,
    melspec_frequency=25,
    frame_length_secs=0.08,
    melspec_depth=160,
    lower_edge_hertz=60.0,
    upper_edge_hertz=7920.0,
    log_floor=1e-2,
    log_offset=0.0,
    logmel_scalar=0.1,
    pcen_alpha=0.5,
    pcen_s=0.1,
    pcen_beta=0.5,
    pcen_delta=2.0,
    pcen_floor=1e-6,
    scaling='log',
    batched=False,
):
  """Convert audio to melspectrogram, using params."""
  # Add front padding so that mel window aligns with audio frame.
  frame_step = int(sample_frequency / melspec_frequency)
  frame_length = int(sample_frequency * frame_length_secs)

  if not batched:
    # Prepare shape for stft operation.
    audio = tf.expand_dims(audio, 0)

  num_padded_samples = int(0.5 * (frame_length - frame_step))
  if num_padded_samples > 0:
    padding = tf.zeros([tf.shape(audio)[0], num_padded_samples], audio.dtype)
    audio = tf.concat([padding, audio], axis=1)

  # stfts is a complex64 Tensor representing the Short-time Fourier Transform
  # of audio. Its shape is [1, ?, num_spectrogram_bins]
  stfts = tf.signal.stft(
      audio, frame_length=frame_length, frame_step=frame_step, pad_end=True
  )

  # An energy spectrogram is the magnitude of the complex-valued STFT.
  # A float32 Tensor of shape [batch_size, ?, num_spectrogram_bins].
  magnitude_spectrograms = tf.abs(stfts)
  num_spectrogram_bins = tf.shape(magnitude_spectrograms)[-1]

  # Warp the linear-scale magnitude spectrograms into the mel-scale.
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      melspec_depth,
      num_spectrogram_bins,
      sample_frequency,
      lower_edge_hertz,
      upper_edge_hertz,
  )
  mel_spectrograms = tf.tensordot(
      magnitude_spectrograms, linear_to_mel_weight_matrix, 1
  )
  mel_spectrograms.set_shape(
      magnitude_spectrograms.shape[:-1].concatenate(
          linear_to_mel_weight_matrix.shape[-1:]
      )
  )

  if scaling == 'log':
    # Mimics the stabilized log used in mel_utils:
    # np.log(np.maximum(data, floor) + additive_offset)
    x = tf.log(tf.maximum(mel_spectrograms, log_floor) + log_offset)
    x = logmel_scalar * x
  elif scaling == 'pcen':
    x = pcen_ops.fixed_pcen(
        mel_spectrograms,
        alpha=pcen_alpha,
        smooth_coef=pcen_s,
        delta=pcen_delta,
        root=(1.0 / pcen_beta),
        floor=pcen_floor,
    )
  elif scaling == 'raw':
    x = mel_spectrograms
  else:
    raise ValueError('Unrecognized melspectrogram scaling mode.')

  num_frames = tf.shape(audio)[1] // sample_frequency * melspec_frequency + 1
  x = x[:, :num_frames]
  if not batched:
    x = tf.squeeze(x, 0)
  return x


def GetAugmentedMelspec(
    audio, sample_frequency, melspec_params, feature_cleaning, filter_params
):
  """Build melspec, apply freq domain augmentation, then clean-up."""
  batched = len(audio.shape) == 2
  melspec_base = Melspec(
      audio=audio,
      batched=batched,
      sample_frequency=sample_frequency,
      **melspec_params
  )
  if not batched:
    melspec_base = tf.expand_dims(melspec_base, 0)
  if filter_params:
    melspec_base = FilterAugmentMelspec(melspec_base, filter_params)
  melspec_base = CleanupMelspec(melspec_base, feature_cleaning)
  if not batched:
    melspec_base = melspec_base[0]
  return melspec_base


def CleanupMelspec(melspec, feature_cleaning):
  """Apply the chosen melspec cleanup technique."""
  if 'strategy' not in feature_cleaning:
    # Default to doing nothing.
    return melspec
  if feature_cleaning['strategy'] == 'whiten':
    melspec = TwoStageWhitening(
        melspec, thresh=feature_cleaning['clean_thresh']
    )
  elif feature_cleaning['strategy'] == 'denoise':
    melspec = MixtureDenoise(melspec, thresh=feature_cleaning['clean_thresh'])
  elif feature_cleaning['strategy'] == 'softmask':
    raise ValueError('Softmask denoiser was removed.')
  elif feature_cleaning and feature_cleaning['strategy']:
    raise ValueError(
        'Unknown feature cleaning strategy : %s' % feature_cleaning
    )
  return melspec


def FilterAugmentMelspec(melspec, filter_params):
  """Apply filtering augmentation to the melspec batch."""
  if (
      not filter_params
      or not filter_params.strategy
      or filter_params.filter_probability <= 0
  ):
    return melspec

  if filter_params.strategy == 'spec_augment':
    filtered = ApplySpecAugment(melspec, batched=True, **filter_params)
  elif filter_params.strategy == 'random_lowpass':
    filters = LowpassFiltersBatch(tf.shape(melspec)[0], melspec.shape[-1])
    # Add a time dimension for broadcasting.
    filters = tf.expand_dims(filters, 1)
    filtered = melspec * filters
  else:
    raise ValueError(
        'Unknown filter augmentation strategy : %s' % filter_params.strategy
    )

  mask = tf.less_equal(
      tf.random.uniform([tf.shape(melspec)[0]], 0.0, 1.0),
      filter_params.filter_probability,
  )
  mask = tf.cast(tf.expand_dims(tf.expand_dims(mask, 1), 2), melspec.dtype)
  melspec = mask * filtered + (1 - mask) * melspec
  return melspec


def TwoStageWhitening(batched_melspec, thresh=0.5):
  """Remove mean and std from melspec, excluding large signal-like values."""
  feature_mean = tf.expand_dims(tf.math.reduce_mean(batched_melspec, axis=1), 1)
  feature_std = tf.expand_dims(tf.math.reduce_std(batched_melspec, axis=1), 1)

  # Remove extreme outliers, and re-estimate mean and std.
  mask = tf.cast(
      tf.less_equal(
          tf.abs(batched_melspec - feature_mean), thresh * feature_std + 1e-4
      ),
      batched_melspec.dtype,
  )
  # number of non-zero elements per channel.
  denom = tf.math.reduce_sum(mask, axis=1)
  masked_x = mask * batched_melspec
  masked_mean = tf.math.reduce_sum(masked_x, axis=1) / (denom + 1)
  masked_mean = tf.expand_dims(masked_mean, 1)
  masked_std = tf.reduce_sum(
      mask * tf.square(batched_melspec - masked_mean), axis=1
  )
  masked_std = tf.sqrt(masked_std / (denom + 1))
  masked_std = tf.expand_dims(masked_std, 1)
  return (batched_melspec - masked_mean) / (masked_std + 1)


def MixtureDenoise(batched_melspec, thresh=1.5):
  """Denoise melspec using an estimated Gaussian noise distribution.

  Forms a noise estimate by a) estimating mean+std, b) removing extreme
  values, c) re-estimating mean+std for the noise, and then d) classifying
  values in the spectrogram as 'signal' or 'noise' based on likelihood under
  the revised estimate. We then apply a mask to return the signal values.

  Args:
    batched_melspec: Batched melspectrogram with shape [B, T, D]
    thresh: z-score theshold for separating signal from noise. On the first
      pass, we use 2*thresh, and on the second pass we use thresh directly.

  Returns:
    Batch of denoised melspectrograms.
  """

  x = batched_melspec
  feature_mean = tf.expand_dims(tf.math.reduce_mean(x, axis=1), 1)
  feature_std = tf.expand_dims(tf.math.reduce_std(x, axis=1), 1)
  demeaned = x - feature_mean
  is_signal = tf.greater_equal(demeaned, 2 * thresh * feature_std)
  is_signal = tf.cast(is_signal, x.dtype)
  is_noise = 1.0 - is_signal

  noise_counts = tf.reduce_sum(is_noise, axis=1)
  noise_mean = tf.math.reduce_sum(x * is_noise, axis=1) / (noise_counts + 1)
  noise_mean = tf.expand_dims(noise_mean, 1)
  noise_var = tf.reduce_sum(is_noise * tf.square(x - noise_mean), axis=1)
  noise_std = tf.sqrt(noise_var / (noise_counts + 1))
  noise_std = tf.expand_dims(noise_std, 1)

  # Recompute signal/noise separation.
  demeaned = x - noise_mean
  is_signal = tf.greater_equal(demeaned, thresh * noise_std)
  is_signal = tf.cast(is_signal, x.dtype)
  is_noise = 1.0 - is_signal

  signal_part = is_signal * x
  noise_part = is_noise * noise_mean
  reconstructed = signal_part + noise_part - noise_mean
  return reconstructed


def FindPeaks(summed_spectral_magnitudes, stft_fps):
  """Locate peaks inside signal of summed spectral magnitudes.

  Args:
    summed_spectral_magnitudes: List of summed spectral components.
    stft_fps: Number of summed magnitude bins per second. Calculated from the
      original sample of the waveform.

  Returns:
    List of filtered peak indices in the array of summed spectral magnitudes.
  """
  threshold = np.mean(summed_spectral_magnitudes) * 1.5
  min_width = int(round(0.5 * stft_fps))
  max_width = int(round(2 * stft_fps))
  width_step_size = int(round((max_width - min_width) / 10))
  widths = range(min_width, max_width, width_step_size)
  peaks = signal.find_peaks_cwt(summed_spectral_magnitudes, widths)
  margin_frames = int(round(0.3 * stft_fps))
  filt_peaks = []
  for x in peaks:
    passing = [
        y >= threshold
        for y in summed_spectral_magnitudes[
            x - margin_frames : x + margin_frames
        ]
    ]
    if any(passing):
      filt_peaks.append(x)
  return filt_peaks


def FindPeaksFromAudio(audio, sample_rate_hz, max_peaks=-1):
  """Construct melspec and find peaks."""
  melspec_rate_hz = 100
  audio = np.float32(audio)
  with tf.Graph().as_default():
    with tf.device('cpu:0'):
      melspec = Melspec(
          audio=audio,
          batched=False,
          sample_frequency=sample_rate_hz,
          melspec_frequency=melspec_rate_hz,
          upper_edge_hertz=10000.0,
      )
      melspec = tf.expand_dims(melspec, 0)
      melspec = MixtureDenoise(melspec, 0.75)[0]

      melspec = tf.Session().run(melspec)
  peaks = FindPeaks(np.sum(melspec, axis=1), melspec_rate_hz)
  peak_energies = np.sum(melspec, axis=1)[peaks]

  def TMelToTAu(tm):
    return 1.0 * tm * sample_rate_hz / melspec_rate_hz

  peaks = [TMelToTAu(p) for p in peaks]

  peak_set = sorted(zip(peak_energies, peaks), reverse=True)
  if max_peaks > 0 and len(peaks) > max_peaks:
    peak_set = peak_set[:max_peaks]
  peaks = [p[1] for p in peak_set]
  return peaks


def MidpointToInterval(midpoint, length_t, min_t, max_t):
  """Find start and endpoints for interval, given a desired midpoint."""
  left_endpoint = midpoint - length_t / 2
  right_endpoint = midpoint + length_t / 2

  # Shift endpoints to try to make the interval length_t, if possible.
  right_overhang = max(right_endpoint - max_t, 0)
  left_endpoint -= right_overhang
  left_overhang = max(min_t - left_endpoint, 0)
  right_endpoint += left_overhang

  left_endpoint = int(max(min_t, left_endpoint))
  right_endpoint = int(min(max_t, right_endpoint))
  return (left_endpoint, right_endpoint)


def SlicePeakedAudio(audio, sample_rate_hz, interval_s=2, max_intervals=5):
  """Extract audio intervals from melspec peaks."""
  if audio.shape[0] <= interval_s * sample_rate_hz:
    return {(0, audio.shape[0]): audio}

  peaks = FindPeaksFromAudio(audio, sample_rate_hz, max_intervals)
  interval_samples = int(interval_s * sample_rate_hz)
  intervals = {
      MidpointToInterval(p, interval_samples, 0, audio.shape[0]) for p in peaks
  }
  intervals = {(a, b): audio[a:b] for (a, b) in intervals}
  return intervals


def LowpassFiltersBatch(batch_size=64, channels=160):
  """Create a batch of random low-pass rolloff frequency envelopes."""
  slopes = tf.random_uniform([batch_size, 1], minval=2, maxval=8)
  offsets = tf.random_uniform([batch_size, 1], minval=0, maxval=5)
  xspace = tf.expand_dims(tf.linspace(0.0, 1.0, channels), 0)
  xspace = tf.tile(xspace, [batch_size, 1])
  envelopes = 1 - 0.5 * (tf.tanh(slopes * (xspace - 0.5) - offsets) + 1)
  return envelopes


def ApplySpecAugmentMask(target, axis, min_length=0.0, max_length=0.5):
  """Generate 0/1 mask."""
  batch_size = tf.shape(target)[0]
  dtype = target.dtype
  masked_portion = tf.random.uniform(
      [batch_size], minval=min_length, maxval=max_length, dtype=dtype
  )
  mask_length = tf.cast(
      masked_portion * tf.cast(target.shape[axis], tf.float32), tf.int64
  )

  diag = tf.range(target.shape.as_list()[axis], dtype=tf.int64)
  diag = tf.expand_dims(diag, 0)
  diag = tf.tile(diag, [batch_size, 1])
  mask = tf.greater_equal(diag, tf.expand_dims(mask_length, 1))
  mask = tf.cast(mask, dtype)

  # Roll each batch element randomly...
  # pylint: disable=g-long-lambda
  def RandRoll(x):
    return tf.roll(
        x,
        tf.random.uniform(
            [], minval=0, maxval=target.shape[axis], dtype=tf.int64
        ),
        axis=0,
    )

  mask = tf.map_fn(RandRoll, mask, dtype=mask.dtype, back_prop=False)
  if axis == 1:
    mask = tf.expand_dims(mask, axis=2)
  else:
    mask = tf.expand_dims(mask, axis=1)
  masked = mask * target
  return masked


def ApplySpecAugment(raw_melspec, batched=False, **kwargs):
  """Apply spectral augmentations."""
  if not batched:
    melspec = tf.expand_dims(raw_melspec, 0)
  else:
    melspec = raw_melspec
  if 'specaug_max_freq_mask' in kwargs:
    max_length = kwargs['specaug_max_freq_mask']
  else:
    max_length = None
  melspec = ApplySpecAugmentMask(
      melspec, 2, min_length=0.0, max_length=max_length
  )
  melspec = ApplySpecAugmentMask(
      melspec, 1, min_length=0.0, max_length=max_length
  )
  if batched:
    return melspec
  else:
    return melspec[0, :, :]
