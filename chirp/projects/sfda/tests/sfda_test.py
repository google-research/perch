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

import shutil
import tempfile

from chirp import config_utils
from chirp.data import pipeline
from chirp.models import frontend
from chirp.projects.sfda import adapt
from chirp.projects.sfda import model_utils
from chirp.projects.sfda import models
from chirp.projects.sfda.configs import audio_baseline
from chirp.projects.sfda.configs import config_globals
from chirp.projects.sfda.configs import image_baseline
from chirp.projects.sfda.configs import tent as tent_config
from chirp.projects.sfda.tests import fake_image_dataset
from chirp.tests import fake_dataset
from flax import traverse_util
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
    self.adapt_dir = tempfile.mkdtemp()
    self.audio_data_dir = tempfile.mkdtemp()
    self.image_data_dir = tempfile.mkdtemp()
    fake_audio_builder = fake_dataset.FakeDataset(data_dir=self.audio_data_dir)
    fake_image_builder = fake_image_dataset.FakeImageDataset(
        data_dir=self.image_data_dir)
    fake_audio_builder.download_and_prepare()
    fake_image_builder.download_and_prepare()
    self.image_builder = fake_image_builder
    self.audio_builder = fake_audio_builder

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.adapt_dir)
    shutil.rmtree(self.audio_data_dir)
    shutil.rmtree(self.image_data_dir)

  def _get_datasets(self, config, modality: adapt.Modality):
    if modality == adapt.Modality.AUDIO:
      adaptation_dataset, _ = pipeline.get_dataset(
          "train[:2]",
          dataset_directory=self.audio_builder.data_dir,
          pipeline=config.adaptation_data_config.pipeline)
      val_dataset, _ = pipeline.get_dataset(
          "train[2:4]",
          dataset_directory=self.audio_builder.data_dir,
          pipeline=config.eval_data_config.pipeline)
    else:
      input_pipeline = models.MODEL_REGISTRY[config.model_config.encoder](
          num_classes=1).get_input_pipeline
      dataset = input_pipeline(data_builder=self.image_builder, split="train")
      dataset = dataset.batch(
          1, drop_remainder=False).batch(
              1, drop_remainder=False)
      adaptation_dataset = val_dataset = dataset
    return adaptation_dataset, val_dataset

  def _get_configs(self,
                   modality: adapt.Modality,
                   use_constant_encoder: bool = True):
    """Create configuration dictionary for training."""
    if modality == adapt.Modality.AUDIO:
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
      if use_constant_encoder:
        config.model_config.encoder = ConstantEncoder(output_dim=32)

      config.model_config.frontend = frontend.MelSpectrogram(
          features=32,
          stride=config.sample_rate_hz // 25,
          kernel_size=10,
          sample_rate=config.sample_rate_hz,
          freq_range=(60, 10_000))
    elif modality == adapt.Modality.IMAGE:
      config = image_baseline.get_config()
      config = config_utils.parse_config(config, config_globals.get_globals())
      config.init_config.target_class_list = "fake_image_dataset"
      config.init_config.input_shape = None
      if use_constant_encoder:
        config.model_config.encoder = models.ImageModelName.CONSTANT
    method_configs = {}
    for method in ["tent"]:
      method_config = _unparsed_configs[method].get_config()
      method_config = config_utils.parse_config(method_config,
                                                config_globals.get_globals())
      method_configs[method] = method_config
    return config, method_configs

  @parameterized.named_parameters(
      ("image", "tent", adapt.Modality.IMAGE),
      ("audio", "tent", adapt.Modality.AUDIO),
  )
  def test_adapt_one_epoch(self, method, modality: adapt.Modality):
    """Test an epoch of adaptation for SFDA methods."""

    # Recover the configurations dict.
    config, method_configs = self._get_configs(modality)
    method_config = method_configs[method]
    sfda_method = method_config.sfda_method
    method_config = getattr(method_config, modality.value)
    method_config.num_epochs = 1

    # Get data
    adaptation_dataset, val_dataset = self._get_datasets(config, modality)

    # Initialize state and parameters
    model_bundle, adaptation_state, key = sfda_method.initialize(
        model_config=config.model_config,
        rng_seed=config.init_config.rng_seed,
        pretrained=False,
        input_shape=config.init_config.input_shape,
        target_class_list=config.init_config.target_class_list,
        adaptation_iterations=method_config.num_epochs *
        len(adaptation_dataset),
        modality=modality,
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
        modality=modality,
        eval_every=config.eval_every,
        sfda_method=sfda_method,
        **method_config)
    self.assertIsNotNone(new_adaptation_state)

  def test_mask_parameters_audio(self):
    """Testing parameter masking used to restrict trainable parameters."""
    config, _ = self._get_configs(modality=adapt.Modality.AUDIO)
    _, params, _, _ = model_utils.prepare_audio_model(
        model_config=config.model_config,
        optimizer_config=None,
        total_steps=0,
        rng_seed=config.init_config.rng_seed,
        input_shape=config.init_config.input_shape,
        pretrained=False,
        target_class_list=config.init_config.target_class_list)

    self._test_mask_parameters(params)

  @parameterized.named_parameters(
      ("resnet", models.ImageModelName.RESNET),
      ("wideresnet", models.ImageModelName.WIDERESNET))
  def test_mask_parameters_image(self, model: models.ImageModelName):
    """Testing parameter masking used to restrict trainable parameters."""

    config, _ = self._get_configs(modality=adapt.Modality.IMAGE)
    config.model_config.encoder = model
    _, params, _, _ = model_utils.prepare_image_model(
        model_config=config.model_config,
        optimizer_config=None,
        total_steps=1,
        rng_seed=config.init_config.rng_seed,
        pretrained=False,
        input_shape=config.init_config.input_shape,
        target_class_list=config.init_config.target_class_list)
    self._test_mask_parameters(params)

  def _test_mask_parameters(self, params):
    # Test BN masking
    masked_params = model_utils.mask_parameters(params,
                                                model_utils.TrainableParams.BN)
    for p, masked in traverse_util.flatten_dict(masked_params).items():
      if any(["norm" in x.lower() for x in p
             ]) and (any(["scale" in x.lower() for x in p]) or
                     any(["bias" in x.lower() for x in p])):
        self.assertFalse(masked)
      else:
        self.assertTrue(masked)

    # Test no masking
    masked_params = model_utils.mask_parameters(params,
                                                model_utils.TrainableParams.ALL)
    for p, masked in traverse_util.flatten_dict(masked_params).items():
      self.assertFalse(masked)

if __name__ == "__main__":
  absltest.main()
