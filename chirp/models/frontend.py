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

"""Model frontends.

A frontend is the part of the model that transforms a sampled audio signal into
a set of features. This module provides Flax modules that can be used
interchangeably.

For some frontends it also defines inverses (e.g., for separation models).
"""
import dataclasses
from typing import Optional, Tuple, Union

from chirp import audio_utils
from chirp import signal
from flax import linen as nn
from jax import numpy as jnp
from jax import scipy as jsp


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

  Attribute:
    power_spectrogram: If true, take the magnitude of the spectrogram.
  """
  power_spectrogram: bool = True

  @nn.compact
  def __call__(self, inputs):
    # For a real-valued signal the number of frequencies returned is n // 2 + 1
    # so we set the STFT window size to return the correct number of features.
    nfft = nperseg = (self.features - 1) * 2

    _, _, stfts = jsp.signal.stft(
        inputs,
        nperseg=nperseg,
        noverlap=nperseg - self.stride,
        nfft=nfft,
        padded=False)

    # STFT does not use SAME padding (i.e., padding with a total of nperseg -
    # stride). Instead it pads with nperseg // 2 on both sides, so the total
    # amount of padding depends on whether nperseg is even or odd. The final
    # output size is (t + stride - (nperseg % 2)) // stride. In our case nperseg
    # is even, so that means we have t // stride + 1 elements. That is one
    # element too many when the stride is a divisor of the input length.
    if inputs.shape[-1] % self.stride == 0:
      stfts = stfts[..., :-1]

    stfts = jnp.swapaxes(stfts, -1, -2)
    return jnp.abs(stfts) if self.power_spectrogram else stfts


class ISTFT(InverseFrontend):
  """Inverse short-term Fourier transform.

  This module uses a Hann window.

  Attribute:
    use_tf_istft: For exporting to TF Lite, the iSTFT can optionally be done
      using an external call to the TF op.
  """

  @nn.compact
  def __call__(self, inputs):
    nfft = nperseg = (inputs.shape[-1] - 1) * 2
    # The STFT transformation threw away the last time step to match our output
    # shape expectations. We'll just pad it with zeros to get it back.
    inputs = jnp.swapaxes(inputs, -1, -2)
    pad_width = ((0, 0),) * (inputs.ndim - 1) + ((0, 1),)
    inputs = jnp.pad(inputs, pad_width, "edge")
    _, istfts = jsp.signal.istft(
        inputs, nperseg=nperseg, noverlap=nperseg - self.stride, nfft=nfft)
    return istfts


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


ScalingConfig = Union[LogScalingConfig, PCENScalingConfig]


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
    scaling_config: The scaling configuration to use.
  """
  kernel_size: int
  sample_rate: int
  freq_range: Tuple[int, int]
  scaling_config: Optional[ScalingConfig] = None

  @nn.compact
  def __call__(self, inputs):
    # Calculate power spectrogram
    _, _, stfts = jsp.signal.stft(
        inputs,
        nperseg=self.kernel_size,
        noverlap=self.kernel_size - self.stride,
        padded=False)
    # See notes in STFT regarding output size
    if inputs.shape[-1] % self.stride == 0:
      stfts = stfts[..., :-1]
    stfts = jnp.swapaxes(stfts, -1, -2)
    magnitude_spectrograms = jnp.abs(stfts)

    # Construct mel-spectrogram
    num_spectrogram_bins = magnitude_spectrograms.shape[-1]
    mel_matrix = signal.linear_to_mel_weight_matrix(self.features,
                                                    num_spectrogram_bins,
                                                    self.sample_rate,
                                                    *self.freq_range)
    mel_spectrograms = magnitude_spectrograms @ mel_matrix

    # Apply frequency scaling
    scaling_config = self.scaling_config
    if isinstance(scaling_config, LogScalingConfig):
      x = audio_utils.log_scale(mel_spectrograms,
                                **dataclasses.asdict(scaling_config))
    elif isinstance(scaling_config, PCENScalingConfig):
      kwargs = dataclasses.asdict(scaling_config)
      if kwargs.pop("spcen"):
        init_smoothing_coef = jnp.ones(
            (self.features,)) * scaling_config.smoothing_coef
        kwargs["smoothing_coef"] = self.param("spcen_smoothing_coef",
                                              lambda _: init_smoothing_coef)
      x, _ = audio_utils.pcen(mel_spectrograms, **kwargs)
    elif scaling_config is None:
      x = mel_spectrograms
    else:
      raise ValueError("Unrecognized scaling mode.")

    return x


class LearnedFrontend(Frontend):
  """Learned filters.

  This frontend is a small wrapper around `nn.Conv`. It learns a filter bank
  where the filters are the convolutional kernels.

  Attributes:
    kernel_size: The size of the convolutional filters.
  """
  kernel_size: int

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    output = nn.Conv(
        features=self.features,
        kernel_size=(self.kernel_size,),
        strides=(self.stride,))(
            # Collapse batch dimensions and add a single channel
            jnp.reshape(inputs, (-1,) + inputs.shape[-1:])[..., jnp.newaxis])
    output = jnp.reshape(output, inputs.shape[:-1] + output.shape[-2:])
    return output


class InverseLearnedFrontend(InverseFrontend):
  """Thin wrapper around a Conv1DTranspose.

  A small wrapper around `nn.ConvTranspose`. It learns the inverse of a filter
  bank where the filters are convolutional kernels.

  Attributes:
    kernel_size: The size of the convolutional filters.
  """
  kernel_size: int

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    output = nn.ConvTranspose(
        features=1, kernel_size=(self.kernel_size,), strides=(self.stride,))(
            jnp.reshape(inputs, (-1,) + inputs.shape[-2:]))
    output = jnp.reshape(output, inputs.shape[:-2] + output.shape[-2:])
    return jnp.squeeze(output, -1)
