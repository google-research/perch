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

import tempfile

from chirp import sep_train
from chirp.configs import separator
from chirp.data import pipeline
from chirp.tests import fake_dataset
from clu import checkpoint
from ml_collections import config_dict
from absl.testing import absltest


class TrainSeparationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.train_dir = tempfile.TemporaryDirectory("train_dir").name

    self.data_dir = tempfile.TemporaryDirectory("data_dir").name
    fake_builder = fake_dataset.FakeDataset(
        sample_rate=32_000, data_dir=self.data_dir)
    fake_builder.download_and_prepare()
    self.builder = fake_builder

  def _get_test_dataset(self, parsed_config):
    ds, dataset_info = pipeline.get_dataset(
        "train",
        dataset_directory=self.builder.data_dir,
        batch_size=parsed_config.batch_size,
        window_size_s=parsed_config.data_config.window_size_s,
        min_gain=parsed_config.data_config.min_gain,
        max_gain=parsed_config.data_config.max_gain,
        mixin_prob=0.5)
    return ds, dataset_info

  def _get_test_config(self, use_small_encoder=True) -> config_dict.ConfigDict:
    """Create configuration dictionary for training."""
    config = separator.get_config()
    with config.unlocked():
      config.batch_size = 2
      config.data_config.window_size_s = 1

      config.train_config.num_train_steps = 1
      config.train_config.checkpoint_every_steps = 1
      config.train_config.log_every_steps = 1
      config.eval_config.eval_steps_per_loop = 1

      if use_small_encoder:
        config.mask_generator_config.base_filters = 2
        config.mask_generator_config.bottleneck_filters = 4
        config.mask_generator_config.output_filters = 8
        config.mask_generator_config.strides = (2, 2)
        config.mask_generator_config.feature_mults = (2, 2)
        config.mask_generator_config.groups = (1, 2)

    return config

  def test_init_baseline(self):
    # Ensure that we can initialize the model with the baseline config.
    config = separator.get_config()
    _, dataset_info = self._get_test_dataset(config)

    model_bundle, train_state = sep_train.initialize_model(
        config,
        dataset_info,
        workdir=self.train_dir,
        rng_seed=config.rng_seed,
        learning_rate=config.learning_rate)
    self.assertIsNotNone(model_bundle)
    self.assertIsNotNone(train_state)

  def test_train_one_step(self):
    config = self._get_test_config(use_small_encoder=True)
    ds, dataset_info = self._get_test_dataset(config)
    model_bundle, train_state = sep_train.initialize_model(
        config,
        dataset_info,
        workdir=self.train_dir,
        rng_seed=config.rng_seed,
        learning_rate=config.learning_rate)

    sep_train.train(
        model_bundle=model_bundle,
        train_state=train_state,
        train_dataset=ds,
        logdir=self.train_dir,
        **config.train_config)
    ckpt = checkpoint.MultihostCheckpoint(self.train_dir)
    self.assertIsNotNone(ckpt.latest_checkpoint)

  def test_eval_one_step(self):
    config = self._get_test_config(use_small_encoder=True)
    ds, dataset_info = self._get_test_dataset(config)
    model_bundle, train_state = sep_train.initialize_model(
        config,
        dataset_info,
        workdir=self.train_dir,
        rng_seed=config.rng_seed,
        learning_rate=config.learning_rate)
    # Write a chekcpoint, or else the eval will hang.
    model_bundle.ckpt.save(train_state)

    sep_train.evaluate_loop(
        model_bundle=model_bundle,
        train_state=train_state,
        valid_dataset=ds,
        workdir=self.train_dir,
        logdir=self.train_dir,
        num_train_steps=0,
        eval_sleep_s=0,
        **config.eval_config)
    ckpt = checkpoint.MultihostCheckpoint(self.train_dir)
    self.assertIsNotNone(ckpt.latest_checkpoint)


if __name__ == "__main__":
  absltest.main()
