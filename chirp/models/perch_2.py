# coding=utf-8
# Copyright 2026 The Perch Authors.
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

"""Key architectural components of Perch 2.0.

These are provided as a reference implementation, only. The various
classification head implementations can be found in `heads.py`.
"""

import functools
import math
import sys

from chirp.models import efficientnet
from chirp.models import frontend as frontend_
from chirp.models import heads
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from scipy import stats


class GeneralizedMixup:
  """Reference implementation of MixUp for multiple source signals.

  A beta-binomial distribution is used to determine the number of signals to
  mix. The support of the distribution is determined by `max_num_signals` and
  the shape by the given alpha and beta parameters. If both of these are equal
  to 1.0, the number of signals will be chosen uniformly. By choosing alpha >
  beta > 1.0 or beta > alpha > 1.0, the distribution will be unimodal and
  skewed towards a larger/smaller numbers of signals. Note that the expected
  number of mixed signals is given by (max_num_signals - 1) * alpha / (alpha +
  beta) + 1

  The signals are mixed using weights drawn from a Dirichlet distribution. The
  concentration parameter of the Dirichlet distribution is given by
  `dirichlet_concentration`. Larger values will result in a more uniform set
  of weights.

  The mixed signal is normalized by the square root of the sum of the squares
  of the weights so that the expected gain is unchanged.

  For this reference implementation, we assume the parent dataset is an indexed
  sequence of examples, like:
  ```
  {index: {
      'audio': audio,
      'label': label,
      'index': index
  }}
  ```
  """

  def __init__(
      self,
      parent_dataset,
      max_num_signals: int = 2,
      beta_binomial_params: tuple[float, float] = (1.0, 1.0),
      dirichlet_concentration: jnp.float32 = jnp.float32(sys.maxsize),
      weigh_labels: bool = False,
      salty_seed: int = 0,
  ):
    """Initialize the MixUp dataset.

    Args:
      parent_dataset: The parent dataset.
      max_num_signals: The maximum number of signals to mix.
      beta_binomial_params: The alpha and beta parameters of the beta-binomial
        distribution.
      dirichlet_concentration: The concentration parameter of the Dirichlet
        distribution.
      weigh_labels: Whether to weigh the labels, i.e., the label of the mixed
        signal is the weighted average of the labels of the input signals. If
        false, the labels are simply merged into a multi-hot vector.
      salty_seed: A salt value used when sampling from the parent dataset.
    """
    self.parent_dataset = parent_dataset
    self._dist = stats.betabinom(max_num_signals - 1, *beta_binomial_params)
    self.dirichlet_concentration = dirichlet_concentration
    self.weigh_labels = weigh_labels
    self.salty_seed = salty_seed

  def __len__(self) -> int:
    return sys.maxsize

  def __getitem__(self, index):
    rng = np.random.default_rng(self.salty_seed + index)
    num_signals = self._dist.rvs(random_state=rng) + 1
    signals = [
        self.parent_dataset[index]
        for index in rng.integers(len(self.parent_dataset), size=num_signals)
    ]
    signals = [signal for signal in signals if signal is not None]
    if not signals:
      return None

    # Mix the audio
    weights = stats.dirichlet.rvs(
        np.full(len(signals), self.dirichlet_concentration), random_state=rng
    )[0].astype(signals[0]['audio'].dtype)
    audio = _mix(weights, [signal['audio'] for signal in signals])
    labels = {}
    for k in ('label', 'index'):
      if k not in signals[0]:
        continue
      signals_k = [signal[k] for signal in signals]
      if self.weigh_labels:
        for signal_, weight in zip(signals_k, weights, strict=True):
          signal_.data *= weight
        labels[k] = sum(signal[k] for signal in signals)
      else:
        labels[k] = functools.reduce(lambda x, y: x.maximum(y), signals_k)
    return {**labels, 'audio': audio, 'index': index}


@jax.jit
def _mix(weights, signals):
  return sum(
      signal * weight for signal, weight in zip(signals, weights)
  ) / jnp.sqrt(jnp.sum(weights**2))


class EmbeddingModel(nn.Module):
  """Embedding model.

  The core of our embedding model: An audio frontend converts raw audio into
  some two-dimensional representation (usually a spectrogram). This is then
  passed to a backbone model which produces a high-dimensional embedding.

  Attributes:
    frontend: A `frontend.Frontend` module that turns raw audio into some
      two-dimensional representation (usually a spectrogram).
    magnitude_scaling: A `frontend.MagnitudeScaling` instance that applies
      magnitude scaling to the output of the frontend.
    embedding_model: The model used to embed the spectrograms. Is assumed to
      take a `train` argument which signifies whether the model is in training
      mode or not.
  """

  frontend: frontend_.Frontend = frontend_.MelSpectrogram(
      features=128,
      stride=32_000 // 100,
      kernel_size=640,
      sample_rate=32_000,
      freq_range=(60, 16_000),
      power=1.0,
      scaling_config=frontend_.LogScalingConfig(),
      nfft=2 ** math.ceil(math.log2(640)),
  )
  backbone: nn.Module = efficientnet.EfficientNet(
      efficientnet.EfficientNetModel.B3,
      include_top=False,
  )

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, train: bool, sow: bool = True
  ) -> jnp.ndarray:
    # Frontend
    unscaled_spectrogram = self.frontend(inputs)
    spectrogram = self.magnitude_scaling(unscaled_spectrogram)

    # Embedding
    spatial_embedding = self.backbone(
        jnp.expand_dims(spectrogram, axis=-1), train=train
    )
    avg_embedding = jnp.mean(spatial_embedding, axis=(-2, -3))
    if sow:
      self.sow('intermediates', 'spectrogram', spectrogram)
      self.sow('intermediates', 'embedding', avg_embedding)
      self.sow('intermediates', 'spatial_embedding', spatial_embedding)

    return spatial_embedding


class MultiHeadClassifier(nn.Module):
  """A multi-head classifier model.

  A classifier which simply applies linear transformations to the output of an
  embedding model.

  Attributes:
    head: The model used to produce the logits.
    embedding_model: The model used to embed the audio.
    stop_gradient_heads: Heads in this set will have a stop_gradient applied to
      the embedding before feeding it to the head.
    train_only_heads: Heads in this set will only be constructed in training
      mode.
    noisy_student_heads: A list of heads which will be used as Noisy Student
      teachers. If non-empty, the model will be run again in inference mode, and
      the outputs of the indicated heads will have a stop-gradient and sigmoid
      applied (to enable use as soft labels with cross-entropy losses). The keys
      for these outputs are prefixed with `teacher_`.
  """

  heads: dict[str, heads.ClassifierHead]
  embedding_model: nn.Module = EmbeddingModel()
  stop_gradient_heads: tuple[str, ...] = ()
  train_only_heads: tuple[str, ...] = ()
  noisy_student_heads: tuple[str, ...] = ()

  def _call_model(
      self, inputs: jnp.ndarray, train: bool, sow: bool
  ) -> dict[str, jnp.ndarray]:
    spatial_embedding = self.embedding_model(inputs, train=train, sow=sow)

    # Classification
    outputs = {}
    for name, head in self.heads.items():
      if name in self.train_only_heads and not train:
        continue
      if name in self.stop_gradient_heads:
        embedding = jax.lax.stop_gradient(spatial_embedding)
      else:
        embedding = spatial_embedding
      logits = head(embedding, train=train)
      outputs[name] = logits
    return outputs

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, train: bool
  ) -> dict[str, jnp.ndarray]:
    # Classification
    outputs = self._call_model(inputs, train=train, sow=True)
    if train and self.noisy_student_heads:
      teacher_outputs = self._call_model(inputs, train=False, sow=False)
      for name in self.noisy_student_heads:
        outputs[f'teacher_{name}'] = jax.lax.stop_gradient(
            jax.nn.sigmoid(teacher_outputs[name])
        )
    return outputs
