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

"""Audio utilities.

General utilities for processing audio and spectrograms.
"""
import concurrent
import itertools
import logging
import os
import tempfile
from typing import Callable, Generator, Sequence
import warnings

from chirp import path_utils
from chirp import signal
from etils import epath
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import scipy as jsp
import librosa
import numpy as np
import requests
from scipy import signal as scipy_signal
import soundfile
import tensorflow as tf

_WINDOW_FNS = {
    'hann': tf.signal.hann_window,
    'hamming': tf.signal.hamming_window,
}
_BOUNDARY_TO_PADDING_MODE = {'zeros': 'CONSTANT'}


def load_audio(
    path: epath.PathLike, target_sample_rate: int, **kwargs
) -> jnp.ndarray:
  """Load a general audio resource."""
  path = os.fspath(path)
  if path.startswith('xc'):
    return load_xc_audio(path, target_sample_rate)
  elif path.startswith('http'):
    return load_url_audio(path, target_sample_rate)
  else:
    return load_audio_file(path, target_sample_rate, **kwargs)


def load_audio_file(
    filepath: str | epath.Path,
    target_sample_rate: int,
    resampling_type: str = 'polyphase',
) -> jnp.ndarray:
  """Read an audio file, and resample it using librosa."""
  filepath = epath.Path(filepath)
  if target_sample_rate <= 0:
    # Use the native sample rate.
    target_sample_rate = None
  extension = os.path.splitext(filepath)[-1].lower()
  if extension in ('wav', 'flac', 'ogg', 'opus'):
    with filepath.open('rb') as f:
      sf = soundfile.SoundFile(file=f)
      audio = sf.read()
      if target_sample_rate is not None:
        audio = librosa.resample(
            y=audio,
            orig_sr=sf.samplerate,
            target_sr=target_sample_rate,
            res_type=resampling_type,
        )
      return audio

  # Handle other audio formats.
  # Because librosa passes file handles to soundfile, we need to copy the file
  # to a temporary file before passing it to librosa.
  with tempfile.NamedTemporaryFile(mode='w+b', suffix=extension) as f:
    with filepath.open('rb') as sf:
      f.write(sf.read())
    # librosa outputs lots of warnings which we can safely ignore when
    # processing all Xeno-Canto files and PySoundFile is unavailable.
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      audio, _ = librosa.load(
          f.name,
          sr=target_sample_rate,
          res_type=resampling_type,
      )
  return audio


def load_audio_window_soundfile(
    filepath: str, offset_s: float, sample_rate: int, window_size_s: float
) -> jnp.ndarray:
  """Load an audio window using Soundfile.

  Args:
    filepath: Path to audio file.
    offset_s: Read offset within the file.
    sample_rate: Sample rate for returned audio.
    window_size_s: Length of audio to read. Reads all if <0.

  Returns:
    Numpy array of loaded audio.
  """
  with epath.Path(filepath).open('rb') as f:
    sf = soundfile.SoundFile(f)
    if offset_s > 0:
      offset = int(offset_s * sf.samplerate)
      sf.seek(offset)
    if window_size_s < 0:
      a = sf.read()
    else:
      window_size = int(window_size_s * sf.samplerate)
      a = sf.read(window_size)
  if len(a.shape) == 2:
    # Downstream ops expect mono audio, so reduce to mono.
    a = a[:, 0]
  if sample_rate > 0:
    a = librosa.resample(
        y=a, orig_sr=sf.samplerate, target_sr=sample_rate, res_type='polyphase'
    )
  return a


def load_audio_window(
    filepath: str,
    offset_s: float,
    sample_rate: int,
    window_size_s: float,
) -> jnp.ndarray:
  """Load a slice of audio from a file, hopefully efficiently."""
  # TODO(tomdenton): Find a reliable way to load a flac audio window.
  # If a flac file has the incorrect length in its header, seeking past the
  # end of the file causes the system to hang. This is a bad enough outcome
  # that we don't risk it.
  try:
    return load_audio_window_soundfile(
        filepath, offset_s, sample_rate, window_size_s
    )
  except soundfile.LibsndfileError:
    logging.info('Failed to load audio with libsndfile: %s', filepath)
  # This fail-over is much slower but more reliable; the entire audio file
  # is loaded (and possibly resampled) and then we extract the target audio.
  audio = load_audio(filepath, sample_rate)
  offset = int(offset_s * sample_rate)
  window_size = int(window_size_s * sample_rate)
  return audio[offset : offset + window_size]


def multi_load_audio_window(
    filepaths: Sequence[str],
    offsets: Sequence[float] | None,
    audio_loader: Callable[[str, float], np.ndarray],
    max_workers: int = 5,
    buffer_size: int = -1,
) -> Generator[np.ndarray, None, None]:
  """Generator for loading audio windows in parallel.

  Note that audio is returned in the same order as the filepaths.
  Also, this ultimately relies on soundfile, which can be buggy in some cases.

  Caution: Because this generator uses an Executor, it can continue holding
  resources while not being used. If you are using this in a notebook, you
  should use this in a 'nameless' context, like:
  ```
  for audio in multi_load_audio_window(...):
    ...
  ```
  or in a try/finally block:
  ```
  audio_iterator = multi_load_audio_window(...)
  try:
    for audio in audio_iterator:
      ...
  finally:
    del(audio_iterator)
  ```
  Otherwise, the generator will continue to hold resources until the notebook
  is closed.

  Args:
    filepaths: Paths to audio to load.
    offsets: Read offset in seconds for each file, or None if no offsets are
      needed.
    audio_loader: Function to load audio given a filepath and offset.
    max_workers: Number of threads to allocate.
    buffer_size: Max number of audio windows to queue up. Defaults to 10x the
      number of workers.

  Yields:
    Loaded audio windows.
  """
  if buffer_size == -1:
    buffer_size = 10 * max_workers
  if offsets is None:
    offsets = [0.0 for _ in filepaths]

  # TODO(tomdenton): Use itertools.batched in Python 3.12+
  def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
      yield batch

  task_iterator = zip(filepaths, offsets)
  batched_iterator = batched(task_iterator, buffer_size)
  mapping = lambda x: audio_loader(x[0], x[1])

  executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
  try:
    yield from itertools.chain.from_iterable(
        executor.map(mapping, batch) for batch in batched_iterator
    )
  finally:
    executor.shutdown(wait=False, cancel_futures=True)


def load_xc_audio(xc_id: str, sample_rate: int) -> jnp.ndarray:
  """Load audio from Xeno-Canto given an ID like 'xc12345'."""
  if not xc_id.startswith('xc'):
    raise ValueError(f'XenoCanto id {xc_id} does not start with "xc".')
  xc_id = xc_id[2:]
  try:
    int(xc_id)
  except ValueError as exc:
    raise ValueError(f'XenoCanto id xc{xc_id} is not an integer.') from exc
  session = requests.Session()
  session.mount(
      'https://',
      requests.adapters.HTTPAdapter(
          max_retries=requests.adapters.Retry(total=5, backoff_factor=0.1)
      ),
  )
  url = f'https://xeno-canto.org/{xc_id}/download'
  try:
    data = session.get(url=url).content
  except requests.exceptions.RequestException as e:
    raise requests.exceptions.RequestException(
        f'Failed to load audio from Xeno-Canto {xc_id}'
    ) from e
  with tempfile.NamedTemporaryFile(suffix='.mp3', mode='wb') as f:
    f.write(data)
    f.flush()
    audio = load_audio_file(f.name, target_sample_rate=sample_rate)
  return audio


def load_url_audio(url: str, sample_rate: int) -> jnp.ndarray:
  """Load audio from a URL."""
  data = requests.get(url).content
  with tempfile.NamedTemporaryFile(mode='wb') as f:
    f.write(data)
    f.flush()
    audio = load_audio_file(f.name, target_sample_rate=sample_rate)
  return audio


# pylint: disable=g-doc-return-or-yield,g-doc-args,unused-argument
def stft_tf(
    x,
    fs=1.0,
    window='hann',
    nperseg=256,
    noverlap=None,
    nfft=None,
    detrend=False,
    return_onesided=True,
    boundary='zeros',
    padded=True,
) -> tf.Tensor:
  """Computes the Short Time Fourier Transform (STFT).

  This is a port of `scipy.signal.stft` to TensorFlow. This allows us to exactly
  reproduce the frontend in the data preprocessing pipeline.
  """

  # Use SciPy's original variable names
  # pylint: disable=invalid-name
  nfft = nperseg if nfft is None else nfft
  noverlap = nperseg // 2 if noverlap is None else noverlap
  nstep = nperseg - noverlap
  if x.dtype.is_complex:
    raise ValueError('tf.signal.stft only supports real signals')
  if window not in _WINDOW_FNS:
    raise ValueError(
        (
            f'tf.signal.stft does not support window {window}, '
            'supported functions are {'
        ),
        '.join(_WINDOW_FNS)}',
    )
  if boundary is not None and boundary not in _BOUNDARY_TO_PADDING_MODE:
    raise ValueError(
        'tf.signal.stft only supports boundary modes None and , '.join(
            _BOUNDARY_TO_PADDING_MODE
        )
    )
  if detrend:
    raise ValueError('tf.signal.stft only supports detrend = False')
  if not return_onesided:
    raise ValueError('tf.signal.stft only supports return_onesided = True')

  input_length = tf.shape(x)[-1]
  # Put the time axis at the end and then put it back
  if boundary in _BOUNDARY_TO_PADDING_MODE:
    mode = _BOUNDARY_TO_PADDING_MODE[boundary]
    paddings = tf.concat(
        [
            tf.repeat([[0, 0]], tf.rank(x) - 1, axis=0),
            [[nperseg // 2, nperseg // 2]],
        ],
        axis=0,
    )
    x = tf.pad(x, paddings, mode)
    input_length += nperseg
  Zxx = tf.signal.stft(
      x,
      frame_length=nperseg,
      frame_step=nstep,
      fft_length=nfft,
      window_fn=_WINDOW_FNS[window],
      pad_end=padded,
  )
  Zxx = tf.linalg.matrix_transpose(Zxx)

  # TODO(bartvm): tf.signal.frame seems to have a bug which sometimes adds
  # too many frames, so we strip those if necessary
  nadd = (-(input_length - nperseg) % nstep) % nperseg if padded else 0
  length = -((input_length + nadd - nperseg + 1) // (noverlap - nperseg))
  Zxx = Zxx[..., :length]

  # Scaling
  Zxx *= 2 / nperseg

  return Zxx


def ema(
    xs: jnp.ndarray,
    gamma: float | jnp.ndarray,
    initial_state: jnp.ndarray | None = None,
    axis: int = 0,
) -> jnp.ndarray:
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
  return ys, final_state  # pytype: disable=bad-return-type  # jax-ndarray


def ema_conv1d(
    xs: jnp.ndarray, gamma: float | jnp.ndarray, conv_width: int
) -> jnp.ndarray:
  """Uses a depth-wise conv1d to approximate the EMA operation."""
  if conv_width == -1:
    conv_width = xs.shape[1]

  left_pad = jnp.repeat(xs[:, 0:1], conv_width - 1, axis=1)
  padded_inp = jnp.concatenate([left_pad, xs], axis=1)

  kernel = jnp.array(
      [(1.0 - gamma) ** (conv_width - 1)]
      + [gamma * (1.0 - gamma) ** k for k in range(conv_width - 2, -1, -1)]
  ).astype(xs.dtype)
  if isinstance(gamma, float) or gamma.ndim == 0:
    kernel = kernel[jnp.newaxis, jnp.newaxis, :]
    kernel = jnp.repeat(kernel, xs.shape[-1], axis=1)
  else:
    kernel = jnp.swapaxes(kernel, 0, 1)
    kernel = kernel[jnp.newaxis, :, :]
  outp = lax.conv_general_dilated(
      padded_inp,
      kernel,
      (1,),
      padding='VALID',
      feature_group_count=xs.shape[-1],
      dimension_numbers=('NTC', 'IOT', 'NTC'),
  )
  return outp


def pcen(
    filterbank_energy: jnp.ndarray,
    smoothing_coef: float = 0.05638943879134889,
    gain: float = 0.98,
    bias: float = 2.0,
    root: float = 2.0,
    eps: float = 1e-6,
    state: jnp.ndarray | None = None,
    conv_width: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Per-Channel Energy Normalization (PCEN).

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
    conv_width: If non-zero, use a convolutional approximation of the EMA, with
      kernel size indicated here. If set to -1, the sequence length will be used
      as the kernel size.

  Returns:
    Filterbank energies with PCEN compression applied (type and shape are
    unchanged). Also returns a state tensor to be used in the next call to
    fixed_pcen.
  """
  if filterbank_energy.ndim < 2:
    raise ValueError('Filterbank energy must have rank >= 2.')

  for name, arr, max_rank in (
      ('gain', gain, 1),
      ('bias', bias, 1),
      ('root', root, 1),
      ('smoothing_coef', smoothing_coef, 1),
      ('eps', eps, 0),
  ):
    if jnp.ndim(arr) > max_rank:
      raise ValueError(f'{name} must have rank at most {max_rank}')

  if conv_width == 0:
    smoothed_energy, filter_state = ema(
        filterbank_energy, smoothing_coef, initial_state=state, axis=-2
    )
  elif len(filterbank_energy.shape) == 3:
    smoothed_energy = ema_conv1d(filterbank_energy, smoothing_coef, conv_width)
    filter_state = None
  else:
    raise ValueError(
        'Can only apply convolutional EMA to inputs with shape [B, T, D].'
    )
  inv_root = 1.0 / root
  pcen_output = (
      filterbank_energy / (eps + smoothed_energy) ** gain + bias
  ) ** inv_root - bias**inv_root

  return pcen_output, filter_state


def log_scale(
    x: jnp.ndarray, floor: float, offset: float, scalar: float
) -> jnp.ndarray:
  """Apply log-scaling.

  Args:
    x: The data to scale.
    floor: Clip input values below this value. This avoids taking the logarithm
      of negative or very small numbers.
    offset: Shift all values by this amount, after clipping. This too avoids
      taking the logarithm of negative or very small numbers.
    scalar: Scale the output by this value.

  Returns:
    The log-scaled data.
  """
  x = jnp.log(jnp.maximum(x, floor) + offset)
  return scalar * x


def random_low_pass_filter(
    key: jnp.ndarray,
    melspec: jnp.ndarray,
    time_axis: int = -2,
    channel_axis: int = -1,
    min_slope: float = 2.0,
    max_slope: float = 8.0,
    min_offset: float = 0.0,
    max_offset: float = 5.0,
) -> jnp.ndarray:
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
      offset_key, shape, minval=min_offset, maxval=max_offset
  )

  shape = [1] * melspec.ndim
  shape[channel_axis] = melspec.shape[channel_axis]
  xspace = jnp.linspace(0.0, 1.0, melspec.shape[channel_axis])
  xspace = jnp.reshape(xspace, shape)

  envelope = 1 - 0.5 * (jnp.tanh(slope * (xspace - 0.5) - offset) + 1)
  return melspec * envelope


def apply_mixture_denoising(
    melspec: jnp.ndarray, threshold: float
) -> jnp.ndarray:
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
  noise_mean = jnp.sum(x * is_noise, axis=0, keepdims=True) / (noise_counts + 1)
  noise_var = jnp.sum(
      is_noise * jnp.square(x - noise_mean), axis=0, keepdims=True
  )
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
    audio = jnp.pad(audio, [[pad_left, pad_right]], mode='wrap')
  return audio


def slice_peaked_audio(
    audio: jnp.ndarray,
    sample_rate_hz: int,
    interval_length_s: float = 6.0,
    max_intervals: int = 5,
) -> jnp.ndarray:
  """Extracts audio intervals from melspec peaks.

  Args:
    audio: input audio sequence of shape [num_samples].
    sample_rate_hz: sample rate of the audio sequence (Hz).
    interval_length_s: length each extracted audio interval.
    max_intervals: upper-bound on the number of audio intervals to extract.

  Returns:
    Sequence of extracted audio intervals, each of shape
    [sample_rate_hz * interval_length_s].
  """
  target_length = int(sample_rate_hz * interval_length_s)

  # Wrap audio to the target length if it's shorter than that.
  audio = pad_to_length_if_shorter(audio, target_length)

  peaks = find_peaks_from_audio(audio, sample_rate_hz, max_intervals)
  left_shift = target_length // 2
  right_shift = target_length - left_shift

  # Ensure that the peak locations are such that
  # `audio[peak - left_shift: peak + right_shift]` is a non-truncated slice.
  peaks = jnp.clip(peaks, left_shift, audio.shape[0] - right_shift)
  # As a result, it's possible that some (start, stop) pairs become identical;
  # eliminate duplicates.
  start_stop = jnp.unique(
      jnp.stack([peaks - left_shift, peaks + right_shift], axis=-1), axis=0
  )

  return start_stop


def find_peaks_from_audio(
    audio: jnp.ndarray,
    sample_rate_hz: int,
    max_peaks: int,
    num_mel_bins: int = 160,
) -> jnp.ndarray:
  """Construct melspec and find peaks.

  Args:
    audio: input audio sequence of shape [num_samples].
    sample_rate_hz: sample rate of the audio sequence (Hz).
    max_peaks: upper-bound on the number of peaks to return.
    num_mel_bins: The number of mel-spectrogram bins to use.

  Returns:
    Sequence of scalar indices for the peaks found in the audio sequence.
  """
  melspec_rate_hz = 100
  frame_length_s = 0.08
  nperseg = int(frame_length_s * sample_rate_hz)
  nstep = sample_rate_hz // melspec_rate_hz
  _, _, spectrogram = jsp.signal.stft(
      audio, nperseg=nperseg, noverlap=nperseg - nstep
  )
  # apply_mixture_denoising/find_peaks_from_melspec expect frequency axis last
  spectrogram = jnp.swapaxes(spectrogram, -1, -2)
  magnitude_spectrogram = jnp.abs(spectrogram)

  # For backwards compatibility, we scale the spectrogram here the same way
  # that the TF spectrogram is scaled. If we don't, the values are too small and
  # end up being clipped by the default configuration of the logarithmic scaling
  magnitude_spectrogram *= nperseg / 2

  # Construct mel-spectrogram
  num_spectrogram_bins = magnitude_spectrogram.shape[-1]
  mel_matrix = signal.linear_to_mel_weight_matrix(
      num_mel_bins,
      num_spectrogram_bins,
      sample_rate_hz,
      lower_edge_hertz=60,
      upper_edge_hertz=10_000,
  )
  mel_spectrograms = magnitude_spectrogram @ mel_matrix

  melspec = log_scale(mel_spectrograms, floor=1e-2, offset=0.0, scalar=0.1)
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
      jnp.arange(min_width, max_width, width_step_size),
  )
  margin_frames = int(round(0.3 * stft_fps))
  start_stop = jnp.clip(
      jnp.stack([peaks - margin_frames, peaks + margin_frames], axis=-1),
      0,
      summed_spectral_magnitudes.shape[0],
  )
  peaks = [
      p
      for p, (a, b) in zip(peaks, start_stop)
      if summed_spectral_magnitudes[a:b].max() >= threshold
  ]
  return jnp.asarray(peaks, dtype=jnp.int32)
