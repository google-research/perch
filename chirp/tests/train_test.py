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

"""Tests for train."""

import os
import tempfile

from chirp import train
from chirp.configs import baseline
from chirp.data import pipeline
from chirp.tests import fake_dataset
from ml_collections import config_dict
import tensorflow as tf
from absl.testing import absltest


class TrainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.train_dir = tempfile.TemporaryDirectory("train_dir").name

    self.data_dir = tempfile.TemporaryDirectory("data_dir").name
    fake_builder = fake_dataset.FakeDataset(data_dir=self.data_dir)
    fake_builder.download_and_prepare()
    self.builder = fake_builder

  def _get_test_config(self) -> config_dict.ConfigDict:
    """Create configuration dictionary for training."""
    config = config_dict.ConfigDict()
    config.batch_size = 1
    config.rng_seed = 0
    config.learning_rate = 0.04
    config.sample_rate_hz = 22050

    train_config = config_dict.ConfigDict()
    train_config.num_train_steps = 1
    train_config.log_every_steps = 1
    train_config.eval_every_steps = 50
    train_config.checkpoint_every_steps = 1
    train_config.tflite_export = True

    eval_config = config_dict.ConfigDict()
    eval_config.eval_steps_per_loop = -1
    eval_config.eval_delay_steps = 500
    eval_config.tflite_export = False

    data_config = config_dict.ConfigDict()
    data_config.window_size_s = 1
    data_config.mixin_prob = 0.0
    data_config.min_gain = 0.15
    data_config.max_gain = 0.75
    data_config.dataset_directory = self.data_dir

    model_config = config_dict.ConfigDict()
    model_config.bandwidth = 0
    model_config.band_stride = 0
    model_config.random_low_pass = False
    model_config.robust_normalization = False
    model_config.encoder_ = "efficientnet-b0"
    model_config.taxonomy_loss_weight = 1.0

    melspec_config = config_dict.ConfigDict()
    melspec_config.melspec_depth = 32
    melspec_config.melspec_frequency = 25
    melspec_config.scaling = "pcen"
    melspec_config.use_tf_stft = True

    config.data_config = data_config
    config.model_config = model_config
    config.model_config.melspec_config = melspec_config
    config.train_config = train_config
    config.eval_config = eval_config
    return config

  def test_export_model(self):
    config = self._get_test_config()
    parsed_config = train.parse_config(config)

    _, dataset_info = pipeline.get_dataset(
        "train",
        dataset_directory=self.builder.data_dir,
        batch_size=parsed_config.batch_size,
        window_size_s=parsed_config.data_config.window_size_s,
        min_gain=parsed_config.data_config.min_gain,
        max_gain=parsed_config.data_config.max_gain,
        mixin_prob=0.5)

    model_bundle, train_state = train.initialize_model(
        dataset_info,
        workdir=self.train_dir,
        data_config=parsed_config.data_config,
        model_config=parsed_config.model_config,
        rng_seed=parsed_config.rng_seed,
        learning_rate=parsed_config.learning_rate)
    input_size = (
        dataset_info.features["audio"].sample_rate *
        parsed_config.data_config.window_size_s)

    train.export_tf_lite(model_bundle, train_state, self.train_dir, input_size)
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(self.train_dir, "model.tflite")))

  def test_init_baseline(self):
    # Ensure that we can initialize the model with the baseline config.
    config = baseline.get_config()
    parsed_config = train.parse_config(config)
    _, dataset_info = pipeline.get_dataset(
        "train",
        dataset_directory=self.builder.data_dir,
        batch_size=parsed_config.batch_size,
        window_size_s=parsed_config.data_config.window_size_s,
        min_gain=parsed_config.data_config.min_gain,
        max_gain=parsed_config.data_config.max_gain,
        mixin_prob=0.5)

    model_bundle, train_state = train.initialize_model(
        dataset_info,
        workdir=self.train_dir,
        data_config=parsed_config.data_config,
        model_config=parsed_config.model_config,
        rng_seed=parsed_config.rng_seed,
        learning_rate=parsed_config.learning_rate)
    self.assertIsNotNone(model_bundle)
    self.assertIsNotNone(train_state)

if __name__ == "__main__":
  absltest.main()
