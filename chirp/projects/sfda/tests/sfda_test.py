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

from chirp import config_utils
from chirp.data import pipeline
from chirp.models import frontend
from chirp.projects.sfda import adapt
from chirp.projects.sfda.configs import audio_baseline
from chirp.projects.sfda.configs import config_globals
from chirp.projects.sfda.configs import tent as tent_config
from chirp.tests import fake_dataset
import flax.linen as nn
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

_unparsed_configs = {"tent": tent_config}


class ConstantEncoder(nn.Module):
  """A no-op encoder for quickly testing adaptation loop."""

  output_dim: int = 32

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      use_running_average: bool  # pylint: disable=redefined-outer-name
  ) -> jnp.ndarray:
    return jnp.zeros([inputs.shape[0], self.output_dim])


class AdaptationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.adapt_dir = tempfile.TemporaryDirectory("adapt_dir").name
    self.data_dir = tempfile.TemporaryDirectory("data_dir").name
    fake_builder = fake_dataset.FakeDataset(data_dir=self.data_dir)
    fake_builder.download_and_prepare()
    self.builder = fake_builder

  def _get_datasets(self, config):
    adaptation_dataset, _ = pipeline.get_dataset(
        "train[:2]",
        dataset_directory=self.builder.data_dir,
        pipeline=config.adaptation_data_config.pipeline)
    val_dataset, _ = pipeline.get_dataset(
        "train[2:4]",
        dataset_directory=self.builder.data_dir,
        pipeline=config.eval_data_config.pipeline)
    return adaptation_dataset, val_dataset

  def _get_configs(self):
    """Create configuration dictionary for training."""
    config = audio_baseline.get_config()
    config = config_utils.parse_config(config, config_globals.get_globals())
    config.init_config.target_class_list = "xenocanto"
    config.sample_rate_hz = 50
    toy_pipeline = pipeline.Pipeline(ops=[
        pipeline.OnlyJaxTypes(),
        pipeline.ConvertBirdTaxonomyLabels(
            source_namespace="ebird2021",
            target_class_list=config.init_config.target_class_list,
            add_taxonomic_labels=True),
        pipeline.Batch(batch_size=2, split_across_devices=True),
        pipeline.RandomSlice(window_size=1),
    ])

    config.adaptation_data_config.pipeline = toy_pipeline
    config.eval_data_config.pipeline = toy_pipeline

    config.model_config.encoder = ConstantEncoder(output_dim=32)

    config.model_config.frontend = frontend.MelSpectrogram(
        features=32,
        stride=config.sample_rate_hz // 25,
        kernel_size=10,
        sample_rate=config.sample_rate_hz,
        freq_range=(60, 10_000))

    method_configs = {}
    for method in ["tent"]:
      method_config = _unparsed_configs[method].get_config()
      method_config = config_utils.parse_config(method_config,
                                                config_globals.get_globals())
      method_configs[method] = method_config
    return config, method_configs

  def test_adapt_one_epoch(self,
                           method: str = "tent",
                           modality: adapt.Modality = adapt.Modality.AUDIO):
    """Test an epoch of adaptation for SFDA methods."""

    # Recover the configurations dict.
    config, method_configs = self._get_configs()
    method_config = method_configs[method]
    sfda_method = method_config.sfda_method
    method_config = getattr(method_config, modality.value)
    method_config.num_epochs = 1

    # Get data
    adaptation_dataset, val_dataset = self._get_datasets(config)

    # Initialize state and parameters
    model_bundle, adaptation_state, key = sfda_method.initialize(
        model_config=config.model_config,
        rng_seed=config.init_config.rng_seed,
        pretrained_ckpt_dir=self.adapt_dir,
        input_shape=config.init_config.input_shape,
        target_class_list=config.init_config.target_class_list,
        adaptation_iterations=method_config.num_epochs *
        len(adaptation_dataset),
        modality=config.modality,
        optimizer_config=method_config.optimizer_config)

    # Perform adaptation.
    new_adaptation_state = adapt.perform_adaptation(
        key=key,
        adaptation_state=adaptation_state,
        adaptation_dataset=adaptation_dataset,
        validation_dataset=val_dataset,
        model_bundle=model_bundle,
        logdir=self.adapt_dir,
        use_supervised_metrics=True,
        target_class_list=config.init_config.target_class_list,
        multi_label=config.multi_label,
        modality=config.modality,
        eval_every=config.eval_every,
        sfda_method=sfda_method,
        **method_config)
    self.assertIsNotNone(new_adaptation_state)


if __name__ == "__main__":
  absltest.main()
