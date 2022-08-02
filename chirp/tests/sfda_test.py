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

"""Tests for adaptation part."""

import tempfile
from typing import Dict, Union

from chirp import adapt
from chirp import config_utils
from chirp.configs import config_globals
from chirp.configs import oracle
from chirp.configs import sfda_baseline
from chirp.configs import source_only
from chirp.data import pipeline
from chirp.models import efficientnet
from chirp.models import frontend
from chirp.tests import fake_dataset
from ml_collections import config_dict

from absl.testing import absltest
from absl.testing import parameterized

_all_configs = {"oracle": oracle, "source_only": source_only}


# from flax import linen as nn
# from jax import numpy as jnp
class AdaptationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.adapt_dir = tempfile.TemporaryDirectory("adapt_dir").name
    self.data_dir = tempfile.TemporaryDirectory("data_dir").name
    fake_builder = fake_dataset.FakeDataset(data_dir=self.data_dir)
    fake_builder.download_and_prepare()
    self.builder = fake_builder

  def _get_test_dataset(self, config):
    ds, dataset_info = pipeline.get_dataset(
        "test",
        dataset_directory=self.builder.data_dir,
        repeat=False,
        shuffle=False,
        pipeline=config.adaptation_data_config.pipeline)
    return ds, dataset_info

  def _get_configs(
      self,
  ) -> Union[config_dict.ConfigDict, Dict[str, config_dict.ConfigDict]]:
    """Create configuration dictionary for training."""
    config = sfda_baseline.get_config()
    config = config_utils.parse_config(config, config_globals.get_globals())

    config.sample_rate_hz = 11_025

    config.adaptation_data_config.pipeline = pipeline.Pipeline(ops=[
        pipeline.MultiHot(),
        pipeline.Batch(batch_size=1, split_across_devices=True),
        pipeline.RandomSlice(window_size=1),
    ])

    config.init_config.model_config.encoder = efficientnet.EfficientNet(
        efficientnet.EfficientNetModel.B0)

    config.init_config.model_config.frontend = frontend.MelSpectrogram(
        features=32,
        stride=32_000 // 25,
        kernel_size=2_560,
        sample_rate=32_000,
        freq_range=(60, 10_000))

    method_configs = {}
    for method in ["source_only", "oracle"]:
      method_config = _all_configs[method].get_config()
      method_config = config_utils.parse_config(method_config,
                                                config_globals.get_globals())
      method_configs[method] = method_config
    return config, method_configs

  @parameterized.parameters(("source_only"), ("oracle"))
  def test_init_method(self, method: str):
    # Ensure that we can initialize the model with the baseline config.
    config, method_configs = self._get_configs()
    _, dataset_info = self._get_test_dataset(config)

    da_method = method_configs[method].da_method
    _, _, _ = da_method.initialize(
        dataset_info,
        config.init_config.model_config,
        config.init_config.rng_seed,
        config.init_config.input_size,
        pretrained_ckpt_dir=self.adapt_dir,
        **method_configs[method])

  @parameterized.parameters(("source_only"), ("oracle"))
  def test_adapt_one_epoch(self, method: str):
    # Ensure that we can initialize the model with the baseline config.
    config, method_configs = self._get_configs()
    dataset, dataset_info = self._get_test_dataset(config)

    da_method = method_configs[method].da_method
    model_bundles, adaptation_state, key = da_method.initialize(
        dataset_info,
        config.init_config.model_config,
        config.init_config.rng_seed,
        config.init_config.input_size,
        pretrained_ckpt_dir=self.adapt_dir,
        **method_configs[method])
    adaptation_state = adapt.perform_adaptation(
        key,
        da_method,
        adaptation_state,
        dataset,
        model_bundles,
        logdir=self.adapt_dir,
        num_epochs=1)


if __name__ == "__main__":
  absltest.main()
