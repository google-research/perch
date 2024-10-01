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

"""Tests for mass-embedding functionality."""

import os
import tempfile

from chirp.projects.zoo import models
from chirp.projects.zoo import taxonomy_model_tf
from chirp.projects.zoo import zoo_interface
from chirp.taxonomy import namespace
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tf_keras

from absl.testing import absltest
from absl.testing import parameterized


class ZooTest(parameterized.TestCase):

  def test_handcrafted_features(self):
    model = models.HandcraftedFeaturesModel.beans_baseline()

    audio = np.zeros([5 * 32000], dtype=np.float32)
    outputs = model.embed(audio)
    # Five frames because we have 5s of audio with window 1.0 and hope 1.0.
    # Beans aggrregation with mfccs creates 20 MFCC channels, and then computes
    # four summary statistics for each, giving a total of 80 output channels.
    self.assertSequenceEqual([5, 1, 80], outputs.embeddings.shape)

  def test_sep_embed_wrapper(self):
    """Check that the joint-model wrapper works as intended."""
    separator = models.PlaceholderModel(
        sample_rate=22050,
        make_embeddings=False,
        make_logits=False,
        make_separated_audio=True,
    )

    embeddor = models.PlaceholderModel(
        sample_rate=22050,
        make_embeddings=True,
        make_logits=True,
        make_separated_audio=False,
    )
    fake_config = config_dict.ConfigDict()
    sep_embed = models.SeparateEmbedModel(
        sample_rate=22050,
        taxonomy_model_tf_config=fake_config,
        separator_model_tf_config=fake_config,
        separation_model=separator,
        embedding_model=embeddor,
    )
    audio = np.zeros(5 * 22050, np.float32)

    outputs = sep_embed.embed(audio)
    # The PlaceholderModel produces one embedding per second, and we have
    # five seconds of audio, with two separated channels, plus the channel
    # for the raw audio.
    # Note that this checks that the sample-rate conversion between the
    # separation model and embedding model has worked correctly.
    self.assertSequenceEqual(
        outputs.embeddings.shape, [5, 3, embeddor.embedding_size]
    )
    # The Sep+Embed model takes the max logits over the channel dimension.
    self.assertSequenceEqual(
        outputs.logits['label'].shape, [5, len(embeddor.class_list.classes)]
    )

  def test_pooled_embeddings(self):
    outputs = zoo_interface.InferenceOutputs(
        embeddings=np.zeros([10, 2, 8]), batched=False
    )
    batched_outputs = zoo_interface.InferenceOutputs(
        embeddings=np.zeros([3, 10, 2, 8]), batched=True
    )

    # Check that no-op is no-op.
    non_pooled = outputs.pooled_embeddings('', '')
    self.assertSequenceEqual(non_pooled.shape, outputs.embeddings.shape)
    batched_non_pooled = batched_outputs.pooled_embeddings('', '')
    self.assertSequenceEqual(
        batched_non_pooled.shape, batched_outputs.embeddings.shape
    )

    for pooling_method in zoo_interface.POOLING_METHODS:
      if pooling_method == 'squeeze':
        # The 'squeeze' pooling method throws an exception if axis size is > 1.
        with self.assertRaises(ValueError):
          outputs.pooled_embeddings(pooling_method, '')
        continue
      elif pooling_method == 'flatten':
        # Concatenates over the target axis.
        time_pooled = outputs.pooled_embeddings(pooling_method, '')
        self.assertSequenceEqual(time_pooled.shape, [2, 80])
        continue

      time_pooled = outputs.pooled_embeddings(pooling_method, '')
      self.assertSequenceEqual(time_pooled.shape, [2, 8])
      batched_time_pooled = batched_outputs.pooled_embeddings(
          pooling_method, ''
      )
      self.assertSequenceEqual(batched_time_pooled.shape, [3, 2, 8])

      channel_pooled = outputs.pooled_embeddings('', pooling_method)
      self.assertSequenceEqual(channel_pooled.shape, [10, 8])
      batched_channel_pooled = batched_outputs.pooled_embeddings(
          '', pooling_method
      )
      self.assertSequenceEqual(batched_channel_pooled.shape, [3, 10, 8])

      both_pooled = outputs.pooled_embeddings(pooling_method, pooling_method)
      self.assertSequenceEqual(both_pooled.shape, [8])
      batched_both_pooled = batched_outputs.pooled_embeddings(
          pooling_method, pooling_method
      )
      self.assertSequenceEqual(batched_both_pooled.shape, [3, 8])

  @parameterized.product(
      model_return_type=('tuple', 'dict'),
      batchable=(True, False),
  )
  def test_taxonomy_model_tf(self, model_return_type, batchable):
    class FakeModelFn:
      output_depths = {'label': 3, 'embedding': 256}

      def infer_tf(self, audio_array):
        outputs = {
            k: np.zeros([audio_array.shape[0], d], dtype=np.float32)
            for k, d in self.output_depths.items()
        }
        if model_return_type == 'tuple':
          # Published Perch models v1 through v4 returned a tuple, not a dict.
          return outputs['label'], outputs['embedding']
        return outputs

    class_list = {
        'label': namespace.ClassList('fake', ['alpha', 'beta', 'delta'])
    }
    wrapped_model = taxonomy_model_tf.TaxonomyModelTF(
        sample_rate=32000,
        model_path='/dev/null',
        window_size_s=5.0,
        hop_size_s=5.0,
        model=FakeModelFn(),
        class_list=class_list,
        batchable=batchable,
    )

    # Check that a single frame of audio is handled properly.
    outputs = wrapped_model.embed(np.zeros([5 * 32000], dtype=np.float32))
    self.assertFalse(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [1, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [1, 3])

    # Check that multi-frame audio is handled properly.
    outputs = wrapped_model.embed(np.zeros([20 * 32000], dtype=np.float32))
    self.assertFalse(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [4, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [4, 3])

    # Check that a batch of single frame of audio is handled properly.
    outputs = wrapped_model.batch_embed(
        np.zeros([10, 5 * 32000], dtype=np.float32)
    )
    self.assertTrue(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [10, 1, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [10, 1, 3])

    # Check that a batch of multi-frame audio is handled properly.
    outputs = wrapped_model.batch_embed(
        np.zeros([2, 20 * 32000], dtype=np.float32)
    )
    self.assertTrue(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [2, 4, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [2, 4, 3])

  def test_whale_model(self):
    # prereq
    class FakeModel(tf_keras.Model):
      """Fake implementation of the humpback_whale SavedModel API.

      The use of `tf_keras` as opposed to `tf.keras` is intentional; the models
      this fakes were exported using "the pure-TensorFlow implementation of
      Keras."
      """

      def __init__(self):
        super().__init__()
        self._sample_rate = 10000
        self._classes = ['Mn']
        self._embedder = tf_keras.layers.Dense(32)
        self._classifier = tf_keras.layers.Dense(len(self._classes))

      def call(self, spectrograms, training=False):
        logits = self.logits(spectrograms)
        return tf.nn.sigmoid(logits)

      @tf.function(
          input_signature=[tf.TensorSpec([None, None, 1], tf.dtypes.float32)]
      )
      def front_end(self, waveform):
        return tf.math.abs(
            tf.signal.stft(
                tf.squeeze(waveform, -1),
                frame_length=1024,
                frame_step=300,
                fft_length=128,
            )[..., 1:]
        )

      @tf.function(
          input_signature=[tf.TensorSpec([None, 128, 64], tf.dtypes.float32)]
      )
      def features(self, spectrogram):
        return self._embedder(tf.math.reduce_mean(spectrogram, axis=-2))

      @tf.function(
          input_signature=[tf.TensorSpec([None, 128, 64], tf.dtypes.float32)]
      )
      def logits(self, spectrogram):
        features = self.features(spectrogram)
        return self._classifier(features)

      @tf.function(
          input_signature=[
              tf.TensorSpec([None, None, 1], tf.dtypes.float32),
              tf.TensorSpec([], tf.dtypes.int64),
          ]
      )
      def score(self, waveform, context_step_samples):
        spectrogram = self.front_end(waveform)
        windows = tf.signal.frame(
            spectrogram, frame_length=128, frame_step=128, axis=1
        )
        shape = tf.shape(windows)
        batch_size = shape[0]
        num_windows = shape[1]
        frame_length = shape[2]
        tf.debugging.assert_equal(frame_length, 128)
        channels_len = shape[3]
        logits = self.logits(
            tf.reshape(
                windows, (batch_size * num_windows, frame_length, channels_len)
            )
        )
        return {'score': tf.nn.sigmoid(logits)}

      @tf.function(input_signature=[])
      def metadata(self):
        return {
            'input_sample_rate': tf.constant(
                self._sample_rate, tf.dtypes.int64
            ),
            'context_width_samples': tf.constant(39124, tf.dtypes.int64),
            'class_names': tf.constant(self._classes),
        }

    # setup
    fake_model = FakeModel()
    batch_size = 2
    duration_seconds = 10
    sample_rate = fake_model.metadata()['input_sample_rate']
    waveform = np.random.randn(
        batch_size,
        sample_rate * duration_seconds,
    )
    expected_frames = int(10 / 3.9124) + 1
    # Call the model to avoid "forward pass of the model is not defined" on
    # save.
    spectrograms = fake_model.front_end(waveform[:, :, np.newaxis])
    fake_model(spectrograms[:, :128, :])
    model_path = os.path.join(tempfile.gettempdir(), 'whale_model')
    fake_model.save(
        model_path,
        signatures={
            'score': fake_model.score,
            'metadata': fake_model.metadata,
            'serving_default': fake_model.score,
            'front_end': fake_model.front_end,
            'features': fake_model.features,
            'logits': fake_model.logits,
        },
    )

    with self.subTest('from_url'):
      # invoke
      model = models.GoogleWhaleModel.load_humpback_model(model_path)
      outputs = model.batch_embed(waveform)

      # verify
      self.assertTrue(outputs.batched)
      self.assertSequenceEqual(
          outputs.embeddings.shape, [batch_size, expected_frames, 1, 32]
      )
      self.assertSequenceEqual(
          outputs.logits['humpback'].shape, [batch_size, expected_frames, 1]
      )

    with self.subTest('from_config'):
      # invoke
      config = config_dict.ConfigDict()
      config.model_url = model_path
      config.sample_rate = float(sample_rate)
      config.window_size_s = 3.9124
      config.peak_norm = 0.02
      model = models.GoogleWhaleModel.from_config(config)
      # Let's check the regular embed this time.
      outputs = model.embed(waveform[0])

      # verify
      self.assertFalse(outputs.batched)
      self.assertSequenceEqual(
          outputs.embeddings.shape, [expected_frames, 1, 32]
      )
      self.assertSequenceEqual(
          outputs.logits['multispecies_whale'].shape, [expected_frames, 1]
      )


if __name__ == '__main__':
  absltest.main()
