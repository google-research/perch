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

"""Model frontends.

A frontend is the part of the model that transforms a sampled audio signal into
a set of features. This module provides Flax modules that can be used
interchangeably.

For some frontends it also defines inverses (e.g., for separation models).
"""
import dataclasses

from chirp import audio_utils
from chirp import signal
from chirp.models import cwt
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import scipy as jsp


@dataclasses.dataclass
class LogScalingConfig:
  """Configuration for log-scaling of mel-spectrogram."""

  floor: float = 1e-2
  offset: float = 0.0
  scalar: float = 0.1


@dataclasses.dataclass
class PCENScalingConfig:
  """Configuration for PCEN normalization of mel-spectrogram."""

  smoothing_coef: float = 0.1
  gain: float = 0.5
  bias: float = 2.0
  root: float = 2.0
  eps: float = 1e-6
  spcen: bool = False
  conv_width: int = 256


ScalingConfig = LogScalingConfig | PCENScalingConfig


def frames_mask(mask: jnp.ndarray, stride: int) -> jnp.ndarray:
  """Converts a mask of samples to a mask of frames.

  Args:
    mask: Array of size (..., time).
    stride: The stride used by the frontend.

  Returns:
    An array of size (..., frames) where frames = ceil(time / stride).
  """
  length = mask.shape[-1]
  num_frames = -(-length // stride)
  pad_width = ((0, 0),) * (mask.ndim - 1) + ((0, num_frames * stride - length),)
  mask = jnp.pad(mask, pad_width)
  frame_masks = jnp.reshape(mask, mask.shape[:-1] + (num_frames, stride))
  return jnp.any(frame_masks, axis=-1)


class Frontend(nn.Module):
  """A audio frontend.

  An audio frontend takes an input of size (..., time) and outputs an array of
  size (..., frames, features) where frames = ceil(time / stride). That is,
  it should behave the same as applying a set of 1D convolutions with `SAME`
  padding.

  Attributes:
    features: The number of features (channels) that the frontend should output.
    stride: The stride to use. For an STFT this is sometimes called the hop
      length.
  """

  features: int
  stride: int

  # TODO(bartvm): Add ScalingConfig with kw_only=True in Python 3.10
  def _magnitude_scale(self, inputs):
    # Apply frequency scaling
    scaling_config = self.scaling_config
    if isinstance(scaling_config, LogScalingConfig):
      outputs = audio_utils.log_scale(
          inputs, **dataclasses.asdict(scaling_config)
      )
    elif isinstance(scaling_config, PCENScalingConfig):
      kwargs = dataclasses.asdict(scaling_config)
      if kwargs.pop("spcen"):
        init_smoothing_coef = (
            jnp.ones((self.features,)) * scaling_config.smoothing_coef
        )
        smoothing_coef = self.param(
            "spcen_smoothing_coef", lambda _: init_smoothing_coef
        )
        smoothing_coef = jnp.clip(smoothing_coef, 0, 1)
        kwargs["smoothing_coef"] = smoothing_coef
      outputs, _ = audio_utils.pcen(inputs, **kwargs)
    elif scaling_config is None:
      outputs = inputs
    else:
      raise ValueError("Unrecognized scaling mode.")

    return outputs


class InverseFrontend(nn.Module):
  """An inverse frontend.

  This takes features of the form (..., frames, features) and outputs
  (..., time), the inverse of the frontend.

  Note that frontends are usually only invertible when the stride is a divisor
  of the input length.

  Attributes:
    stride: The stride that was used for the frontend. This tells the inverse
      frontend how many samples to generate per step.
  """

  stride: int


class STFT(Frontend):
  """Short-term Fourier transform.

  This module uses a Hann window.

  For efficiency, it might be useful to set the number of features to 2^n + 1
  for some non-negative integer n. This will guarantee that the length of the
  FFT is a power of two.

  Note that if magnitude scaling is used, this frontend is no longer invertible.

  Attribute:
    power: If given, calculate the magnitude spectrogram using the given power.
      The default is 2.0 for the power spectrogram. Pass 1.0 to get the energy
      spectrogram. If `None`, then the complex-valued STFT will be returned.
    scaling_config: The magnitude scaling configuration to use.
  """

  power: float | None = 2.0
  scaling_config: ScalingConfig | None = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    if self.power is None and self.scaling_config is not None:
      raise ValueError("magnitude scaling requires a magnitude spectrogram")
    # For a real-valued signal the number of frequencies returned is n // 2 + 1
    # so we set the STFT window size to return the correct number of features.
    nfft = nperseg = (self.features - 1) * 2

    _, _, stfts = jsp.signal.stft(
        inputs,
        nperseg=nperseg,
        noverlap=nperseg - self.stride,
        nfft=nfft,
        padded=False,
    )

    # STFT does not use SAME padding (i.e., padding with a total of nperseg -
    # stride). Instead it pads with nperseg // 2 on both sides, so the total
    # amount of padding depends on whether nperseg is even or odd. The final
    # output size is (t + stride - (nperseg % 2)) // stride. In our case nperseg
    # is even, so that means we have t // stride + 1 elements. That is one
    # element too many when the stride is a divisor of the input length.
    if inputs.shape[-1] % self.stride == 0:
      stfts = stfts[..., :-1]

    stfts = jnp.swapaxes(stfts, -1, -2)
    stfts = jnp.abs(stfts) ** self.power if self.power is not None else stfts
    return self._magnitude_scale(stfts)


class ISTFT(InverseFrontend):
  """Inverse short-term Fourier transform.

  This module uses a Hann window.
  """

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    nfft = nperseg = (inputs.shape[-1] - 1) * 2
    # The STFT transformation threw away the last time step to match our output
    # shape expectations. We'll just pad it with zeros to get it back.
    inputs = jnp.swapaxes(inputs, -1, -2)
    pad_width = ((0, 0),) * (inputs.ndim - 1) + ((0, 1),)
    inputs = jnp.pad(inputs, pad_width, "edge")
    _, istfts = jsp.signal.istft(
        inputs, nperseg=nperseg, noverlap=nperseg - self.stride, nfft=nfft
    )
    return istfts


class MelSpectrogram(Frontend):
  """Mel-spectrogram frontend.

  This frontend begins by calculating the short-term Fourier transform of the
  audio using a Hann window and padding. Next, it constructs a mel-spectrogram:
  It takes the magnitude of the STFT (power spectrogram), maps the frequencies
  to the mel-scale, and bins frequencies together using a series of partially
  overlapping triangle filters.

  Then an optional scaling step is applied, which can be the logarithm (i.e., a
  log power spectrum as used by mel-frequency cepstrums) or PCEN. The smoothing
  coefficients  of PCEN can optionally be learned as is done by the LEAF
  frontend (sPCEN).

  Finally, the last few frames are discarded so that the number of output
  frames is the expected size (i.e., similar to what you would expect when
  doing a set of 1D convolutions with the same kernel size and stride and
  `SAME` padding).

  Attributes:
    kernel_size: The window size to use for the STFT.
    sample_rate: The sampling rate of the inputs. This is used to calculate the
      conversion to mel-scale.
    freq_range: The frequencies to include in the output. Frequencies outside of
      this range are simply discarded.
    scaling_config: The magnitude scaling configuration to use.
  """

  kernel_size: int
  sample_rate: int
  freq_range: tuple[float, float]
  power: float = 2.0
  scaling_config: ScalingConfig | None = None
  nfft: int | None = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    # Calculate power spectrogram
    _, _, stfts = jsp.signal.stft(
        inputs,
        nperseg=self.kernel_size,
        noverlap=self.kernel_size - self.stride,
        nfft=self.nfft,
        padded=False,
    )
    # See notes in STFT regarding output size
    if inputs.shape[-1] % self.stride == 0:
      stfts = stfts[..., :-1]
    stfts = jnp.swapaxes(stfts, -1, -2)
    magnitude_spectrograms = jnp.abs(stfts) ** self.power

    # Construct mel-spectrogram
    num_spectrogram_bins = magnitude_spectrograms.shape[-1]
    mel_matrix = signal.linear_to_mel_weight_matrix(
        self.features, num_spectrogram_bins, self.sample_rate, *self.freq_range
    )
    output = magnitude_spectrograms @ mel_matrix
    return self._magnitude_scale(output)


class SimpleMelspec(Frontend):
  """Minimal RFFT-based Melspec implementation."""

  kernel_size: int
  sample_rate: int
  freq_range: tuple[int, int]
  power: float = 2.0
  scaling_config: ScalingConfig | None = None
  nfft: int | None = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    flat_inputs = jnp.reshape(inputs, (-1,) + inputs.shape[-1:] + (1,))
    # Note that Scipy uses VALID padding, with additional padding logic.
    # As a result, the outputs are numerically inequivalent.
    framed = jax.lax.conv_general_dilated_patches(
        flat_inputs,
        (self.kernel_size,),
        (self.stride,),
        "SAME",
        dimension_numbers=("NTC", "OIT", "NTC"),
    )

    window = jnp.hanning(self.kernel_size)
    # The scipy stft default scaling resolves down to this...
    # For the stft, the scalar is squared then sqrt'ed.
    window *= 1.0 / window.sum()
    windowed = window[jnp.newaxis, jnp.newaxis, :] * framed
    stfts = jnp.fft.rfft(windowed, n=self.nfft, axis=-1)
    mags = stfts.real**2 + stfts.imag**2
    if self.power == 1.0:
      mags = jnp.sqrt(mags)
    elif self.power == 2.0:
      pass
    else:
      mags = mags ** (self.power / 2.0)

    n_bins = mags.shape[-1]
    mel_matrix = signal.linear_to_mel_weight_matrix(
        num_mel_bins=self.features,
        num_spectrogram_bins=n_bins,
        sample_rate=self.sample_rate,
        lower_edge_hertz=self.freq_range[0],
        upper_edge_hertz=self.freq_range[1],
    )
    output = mags @ mel_matrix
    output = jnp.reshape(output, inputs.shape[:-1] + output.shape[-2:])
    return self._magnitude_scale(output)


class MFCC(Frontend):
  """MFC coefficients frontend.

  This frontend begins by calculating the mel-spectrogram of the audio, then
  computes its discrete cosine transform.

  Attributes:
    mel_spectrogram_frontend: Frontend used for computing mel-spectrograms out
      of audio sequences.
    num_coefficients: Number of MFC coefficients to keep.
    aggregate_over_time: If True, aggregate the MFCs (of shape [..., num_frames,
      num_coefficients]) over the time axis using mean, standard deviation, min,
      and max operations. The result is four tensors of shape [...,
      num_coefficients] that are then concatenated into a single output of shape
      [..., 4 * num_coefficients]. This mirrors the processing done in the BEANS
      benchmark (Hagiwara et al., 2022).
  """

  mel_spectrogram_frontend: MelSpectrogram
  num_coefficients: int | None = None
  aggregate_over_time: bool = True

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    mel_spectrograms = self.mel_spectrogram_frontend(inputs, train)
    outputs = jsp.fft.dct(
        mel_spectrograms, type=2, n=self.num_coefficients, axis=-1, norm="ortho"
    )
    if self.aggregate_over_time:
      outputs = jnp.concatenate(
          [
              outputs.mean(axis=-2),
              outputs.std(axis=-2),
              outputs.min(axis=-2),
              outputs.max(axis=-2),
          ],
          axis=-1,
      )

    return outputs


class LearnedFrontend(Frontend):
  """Learned filters.

  This frontend is a small wrapper around `nn.Conv`. It learns a filter bank
  where the filters are the convolutional kernels.

  Note that if magnitude scaling is used, this frontend is no longer invertible.

  Attributes:
    kernel_size: The size of the convolutional filters.
    scaling_config: The magnitude scaling configuration to use.
  """

  kernel_size: int
  scaling_config: ScalingConfig | None = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    output = nn.Conv(
        features=self.features,
        kernel_size=(self.kernel_size,),
        strides=(self.stride,),
    )(
        # Collapse batch dimensions and add a single channel
        jnp.reshape(inputs, (-1,) + inputs.shape[-1:] + (1,))
    )
    output = jnp.reshape(output, inputs.shape[:-1] + output.shape[-2:])
    return self._magnitude_scale(output)


class InverseLearnedFrontend(InverseFrontend):
  """Thin wrapper around a Conv1DTranspose.

  A small wrapper around `nn.ConvTranspose`. It learns the inverse of a filter
  bank where the filters are convolutional kernels.

  Attributes:
    kernel_size: The size of the convolutional filters.
  """

  kernel_size: int

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    output = nn.ConvTranspose(
        features=1, kernel_size=(self.kernel_size,), strides=(self.stride,)
    )(jnp.reshape(inputs, (-1,) + inputs.shape[-2:]))
    output = jnp.reshape(output, inputs.shape[:-2] + output.shape[-2:])
    return jnp.squeeze(output, -1)


class MorletWaveletTransform(Frontend):
  """Morlet wavelet transform.

  The Morlet wavelet transform is a wavelet transformation using Morlet
  wavelets. This is like a short-term Fourier transform with Gaussian windows,
  but where the window size is different for each frequency. This allows for
  arbitrary trade-offs of the time- and frequency resolution.

  Note that technically speaking this module uses Gabor filters instead of
  Morlet wavelets. Gabor filters don't have the constant shift required to make
  them invertible for low frequencies, but in practice this barely matters.

  The LEAF frontend uses this transformation with stride 1 as the first step.
  Like LEAF, we initialize the Gabor filters to resemble a mel-spectrogram with
  the given frequency range.

  Attributes:
    kernel_size: The kernel size to use for the filters.
    sample_rate: The sample rate of the input. Used to interpret the frequency
      range for initilizing the filters.
    freq_range: The filters are initialized to resemble a mel-spectrogram. These
      values determine the minimum and maximum frequencies of those filters.
  """

  kernel_size: int
  sample_rate: int
  freq_range: tuple[int, int]
  power: float = 2.0
  scaling_config: ScalingConfig | None = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    input_signal = jnp.reshape(inputs, (-1,) + inputs.shape[-1:] + (1,))

    params = cwt.melspec_params(
        self.features, self.sample_rate, *self.freq_range
    )
    gabor_mean = self.param("gabor_mean", lambda rng: params[0])
    gabor_std = self.param("gabor_std", lambda rng: params[1])
    sigma = gabor_mean * gabor_std
    gabor_filter = cwt.gabor_filter(
        sigma, cwt.Domain.TIME, cwt.Normalization.L1
    )
    filtered_signal = cwt.convolve_filter(
        gabor_filter,
        input_signal,
        gabor_std,
        cwt.Normalization.L1,
        self.kernel_size,
        stride=(self.stride,),
    )

    power_signal = jnp.abs(filtered_signal) ** self.power

    scaled_signal = self._magnitude_scale(power_signal)

    output = jnp.reshape(
        scaled_signal, inputs.shape[:-1] + scaled_signal.shape[-2:]
    )

    return output
