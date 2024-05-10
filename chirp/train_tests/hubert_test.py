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

"""Tests for HuBERT."""
import tempfile
from typing import Callable
from chirp import config_utils
from chirp.configs import config_globals
from chirp.configs import hubert_base_pq
from chirp.data import utils as data_utils
from chirp.preprocessing import pipeline
from chirp.train import hubert as hubert_train
from chirp.train_tests import fake_dataset
from clu import checkpoint
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random
from ml_collections import config_dict
from absl.testing import absltest


class ConstantEarlyFeatureExtractor(nn.Module):
  """A no-op encoder for quickly testing train+test loops."""

  conv_layer_tuples: tuple[tuple[int, int, int], ...]
  dropout_prob: float = 0.0
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  deprecated_group_conv: bool = False
  sz: int = 2
  csz: int = 1

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, *unused_args, **unused_kwargs
  ) -> jnp.ndarray:
    del unused_args, unused_kwargs
    return jnp.zeros([inputs.shape[0], self.sz, self.csz])


class ConstantLateFeatureExtractor(nn.Module):
  model_dims: int = 6
  sz: int = 2

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, *unused_args, **unused_kwargs
  ) -> jnp.ndarray:
    del unused_args, unused_kwargs
    return [jnp.zeros([inputs.shape[0], self.sz, self.model_dims])]


class HuBERTTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.train_dir = tempfile.TemporaryDirectory("train_dir").name
    self.data_dir = tempfile.TemporaryDirectory("data_dir").name
    fake_builder = fake_dataset.FakeDataset(data_dir=self.data_dir)
    fake_builder.download_and_prepare()
    self.builder = fake_builder

    self.sample_rate_hz = 32000
    self.model_dims = 6
    self.window_size_s = 5

    # Adds only one product quantizer, in the melspec space.
    self.num_centroids = 3
    self.num_sections = 2
    self.readout_points = (0,)
    self.alpha = 1.0

    self.config = self._get_test_config()

    self.initialized_model = hubert_train.initialize_model(
        workdir=self.train_dir,
        num_train_steps=1,
        conv_layer_tuples=tuple([(1, 10, 80000)]),  # this leads to 2 frames.
        early_fs_class=ConstantEarlyFeatureExtractor,
        **self.config.init_config
    )
    (self.model_bundle, self.train_state, self.learn_rate_schedule) = (
        self.initialized_model
    )
    self.model = self.model_bundle.model
    self.key = self.model_bundle.key
    self.model_state = self.train_state.model_state
    self.params = self.train_state.params

  def _get_test_dataset(self, config):
    """Gets the dataset to use for these tests."""
    ds, dataset_info = data_utils.get_dataset(
        "train",
        dataset_directory=self.builder.data_dir,
        pipeline=config.train_dataset_config.pipeline,
    )
    return ds, dataset_info

  def _get_test_config(
      self, config_module=hubert_base_pq
  ) -> config_dict.ConfigDict:
    """Reduces test config sizes to avoid memory blowouts."""
    config = config_module.get_config()

    config.sample_rate_hz = self.sample_rate_hz
    config.train_config.num_train_steps = 1
    config.train_config.log_every_steps = 1
    config.train_config.checkpoint_every_steps = 1
    config.eval_config.eval_steps_per_checkpoint = 1
    config.init_config.base_quantizer_config.num_centroids = self.num_centroids
    config.init_config.quantizer_config.num_sections = self.num_sections
    config.init_config.model_config.readout_points = self.readout_points
    config.init_config.model_config.mask_config.min_masks = 0
    config.init_config.model_config.mask_config.mask_length = 1
    config.init_config.frontend_config.stride = 80000  # yields 2 frames.
    config.init_config.frontend_config.sample_rate = self.sample_rate_hz
    config.init_config.input_size = self.window_size_s * self.sample_rate_hz

    config = config_utils.parse_config(config, config_globals.get_globals())

    config.init_config.model_config.late_feature_extractor = (
        ConstantLateFeatureExtractor()
    )

    config.train_dataset_config.pipeline = pipeline.Pipeline(
        ops=[
            pipeline.OnlyJaxTypes(),
            pipeline.ConvertBirdTaxonomyLabels(
                source_namespace="ebird2021",
                target_class_list="xenocanto",
                add_taxonomic_labels=True,
            ),
            pipeline.MixAudio(mixin_prob=0.0),
            pipeline.Batch(batch_size=2, split_across_devices=True),
            pipeline.RandomSlice(window_size=self.window_size_s),
            pipeline.RandomNormalizeAudio(min_gain=0.15, max_gain=0.25),
        ]
    )

    config.eval_dataset_config.pipeline = pipeline.Pipeline(
        ops=[
            pipeline.OnlyJaxTypes(),
            pipeline.MultiHot(),
            pipeline.Batch(batch_size=2, split_across_devices=True),
            pipeline.Slice(
                window_size=self.window_size_s, start=0.0, names=("audio",)
            ),
            pipeline.NormalizeAudio(target_gain=0.2, names=("audio",)),
        ]
    )

    return config

  def test_shapes(self):
    """Test that the shapes of outputs returned by HuBERT are as expected."""
    batch_size = 2
    num_frames = 2
    inputs = jnp.zeros([batch_size, self.window_size_s * self.sample_rate_hz])
    step_key, key = random.split(self.key)
    dropout_key, low_pass_key = random.split(step_key)
    mask_key, _ = random.split(key)
    variables = {"params": self.params, **self.model_state}
    model_outputs, _ = self.model.apply(
        variables,
        inputs,
        train=True,
        mask_key=mask_key,
        train_mode_quantizer=True,
        mutable=list(self.model_state.keys()),
        rngs={
            "dropout": dropout_key,
            "low_pass": low_pass_key,
        },
    )

    # Ensure that the number of logits matches that of targets. There will be
    # as many "sets" of these as there are quantizers. In this case it should
    # be just one, since `quantizer_points` has a single element.
    self.assertEqual(len(model_outputs.logits), len(model_outputs.targets))
    self.assertLen(model_outputs.logits, 1)

    # Ensure the shapes of embeddings and logits are as expected.
    self.assertSequenceEqual(
        model_outputs.embedding[-1].shape,
        (batch_size, num_frames, self.model_dims),
    )
    self.assertSequenceEqual(
        model_outputs.logits[-1].shape,
        (self.num_sections, batch_size, num_frames, self.num_centroids),
    )

  def test_gradients(self):
    """Test that there is no gradient from HuBERT's loss to the quantizers."""

    batch_size = 2
    inputs = jnp.zeros([batch_size, self.window_size_s * self.sample_rate_hz])
    step_key, key = random.split(self.key)
    dropout_key, low_pass_key = random.split(step_key)
    mask_key, _ = random.split(key)

    def step(params, model_state):
      variables = {"params": params, **model_state}
      model_outputs, _ = self.model.apply(
          variables,
          inputs,
          train=True,
          mask_key=mask_key,
          train_mode_quantizer=True,
          mutable=list(model_state.keys()),
          rngs={
              "dropout": dropout_key,
              "low_pass": low_pass_key,
          },
      )

      hubert_loss = jnp.mean(
          hubert_train.hubert_loss_from_outputs(
              model_outputs, alpha=self.alpha, hubert_loss_mult=1.0
          )
      )
      return hubert_loss

    _, grads = jax.value_and_grad(step)(self.params, self.model_state)
    self.assertIsNotNone(grads)

    def get_all_leaves(d):
      leaves = []
      if not isinstance(d, dict):
        leaves.append(d)
      else:
        for _, v in d.items():
          leaves.extend(get_all_leaves(v))
      return leaves

    for k, v in grads.items():
      if "quantizer" in k:
        quantizer_grads = get_all_leaves(v)
        for quant_grad in quantizer_grads:
          self.assertTrue((quant_grad == jnp.zeros_like(quant_grad)).all())

  def test_train_one_step(self):
    """Test one step of training."""

    ds, _ = self._get_test_dataset(self.config)
    hubert_train.train(
        *self.initialized_model,
        train_dataset=ds,
        reload_quantizer=False,
        logdir=self.train_dir,
        num_train_steps=1,
        log_every_steps=1,
        checkpoint_every_steps=1,
        num_quantizer_pretrain_steps=self.config.train_config.num_quantizer_pretrain_steps,
        quant_loss_mult=0.0,
        readout_loss_mult=0.0,
        hubert_loss_mult=1.0
    )
    ckpt = checkpoint.MultihostCheckpoint(self.train_dir)
    self.assertIsNotNone(ckpt.latest_checkpoint)

  def test_eval_one_step(self):
    ds, _ = self._get_test_dataset(self.config)
    # Write a checkpoint, or else the eval will hang.
    self.model_bundle.ckpt.save(self.train_state)
    self.config.eval_config.num_train_steps = 0
    hubert_train.evaluate(
        model_bundle=self.model_bundle,
        train_state=self.train_state,
        learning_rate_schedule=self.learn_rate_schedule,
        valid_dataset=ds,
        workdir=self.train_dir,
        eval_sleep_s=0,
        **self.config.eval_config
    )
    ckpt = checkpoint.MultihostCheckpoint(self.train_dir)
    self.assertIsNotNone(ckpt.latest_checkpoint)


if __name__ == "__main__":
  absltest.main()
