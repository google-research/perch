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

from chirp import audio_utils
from chirp import config_utils
from chirp import sep_train
from chirp.configs import config_globals
from chirp.configs import separator
from chirp.data import pipeline
from chirp.data.bird_taxonomy import bird_taxonomy
from chirp.tests import fake_dataset
from clu import checkpoint
from ml_collections import config_dict
import tensorflow as tf

from absl.testing import absltest


class TrainSeparationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.train_dir = tempfile.TemporaryDirectory("train_dir").name

    self.data_dir = tempfile.TemporaryDirectory("data_dir").name

    # The following config should be practically equivalent to what was done
    # before: audio feature shape will be [sample_rate]
    config = bird_taxonomy.BirdTaxonomyConfig(
        name="sep_train_test_config",
        sample_rate_hz=32_000,
        localization_fn=audio_utils.slice_peaked_audio,
        interval_length_s=1.0,
    )
    fake_builder = fake_dataset.FakeDataset(
        config=config, data_dir=self.data_dir)
    fake_builder.download_and_prepare()
    self.builder = fake_builder

  def _get_test_dataset(self, split, config):
    config.dataset_directory = self.builder.data_dir
    config.tfds_data_dir = ""
    ds, dataset_info = pipeline.get_dataset(split, **config)
    return ds, dataset_info

  def _get_test_config(self, use_small_encoder=True) -> config_dict.ConfigDict:
    """Create configuration dictionary for training."""
    config = separator.get_config()

    config.train_data_config.batch_size = 2
    config.train_data_config.window_size_s = 1

    config.eval_data_config.batch_size = 2
    config.eval_data_config.window_size_s = 1

    config.train_config.num_train_steps = 1
    config.train_config.checkpoint_every_steps = 1
    config.train_config.log_every_steps = 1
    config.eval_config.eval_steps_per_checkpoint = 1

    if use_small_encoder:
      soundstream_config = config_dict.ConfigDict()
      soundstream_config.base_filters = 2
      soundstream_config.bottleneck_filters = 4
      soundstream_config.output_filters = 8
      soundstream_config.strides = (2, 2)
      soundstream_config.feature_mults = (2, 2)
      soundstream_config.groups = (1, 2)
      config.init_config.model_config.mask_generator = config_utils.callable_config(
          "soundstream_unet.SoundstreamUNet", soundstream_config)

    config = config_utils.parse_config(config, config_globals.get_globals())
    return config

  def test_init_baseline(self):
    # Ensure that we can initialize the model with the baseline config.
    config = separator.get_config()
    config = config_utils.parse_config(config, config_globals.get_globals())

    model_bundle, train_state = sep_train.initialize_model(
        workdir=self.train_dir, **config.init_config)
    self.assertIsNotNone(model_bundle)
    self.assertIsNotNone(train_state)

  def test_train_one_step(self):
    config = self._get_test_config(use_small_encoder=True)
    ds, _ = self._get_test_dataset("train", config.train_data_config)
    model = sep_train.initialize_model(
        workdir=self.train_dir, **config.init_config)

    sep_train.train(
        *model, train_dataset=ds, logdir=self.train_dir, **config.train_config)
    ckpt = checkpoint.MultihostCheckpoint(self.train_dir)
    self.assertIsNotNone(ckpt.latest_checkpoint)

  def test_eval_one_step(self):
    config = self._get_test_config(use_small_encoder=True)
    config.init_config.model_config.mask_generator.groups = (1, 1)
    config.eval_config.num_train_steps = 0

    ds, _ = self._get_test_dataset("test", config.eval_data_config)
    model_bundle, train_state = sep_train.initialize_model(
        workdir=self.train_dir, **config.init_config)
    # Write a chekcpoint, or else the eval will hang.
    model_bundle.ckpt.save(train_state)

    sep_train.evaluate_loop(
        model_bundle=model_bundle,
        train_state=train_state,
        valid_dataset=ds,
        workdir=self.train_dir,
        logdir=self.train_dir,
        eval_sleep_s=0,
        **config.eval_config)
    ckpt = checkpoint.MultihostCheckpoint(self.train_dir)
    self.assertIsNotNone(ckpt.latest_checkpoint)

  def test_export_model(self):
    config = self._get_test_config(use_small_encoder=True)
    config.init_config.model_config.mask_generator.groups = (1, 1)
    model_bundle, train_state = sep_train.initialize_model(
        workdir=self.train_dir, **config.init_config)

    frame_size = 32 * 2 * 2
    sep_train.export_tf(model_bundle, train_state, self.train_dir, frame_size)
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(self.train_dir, "model.tflite")))
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(self.train_dir, "savedmodel/saved_model.pb")))

if __name__ == "__main__":
  absltest.main()
