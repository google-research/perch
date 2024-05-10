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

"""Tests for train."""

import os
import tempfile

from chirp import config_utils
from chirp.configs import baseline
from chirp.configs import baseline_attention
from chirp.configs import baseline_mel_conformer
from chirp.configs import config_globals
from chirp.data import utils as data_utils
from chirp.models import efficientnet
from chirp.models import frontend
from chirp.preprocessing import pipeline
from chirp.taxonomy import namespace
from chirp.train_tests import fake_dataset
from chirp.train import classifier
from clu import checkpoint
from flax import linen as nn
import jax
from jax import numpy as jnp
from ml_collections import config_dict
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized

TEST_WINDOW_S = 1


class ConstantEncoder(nn.Module):
  """A no-op encoder for quickly testing train+test loops."""

  output_dim: int = 32

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,  # pylint: disable=redefined-outer-name
      use_running_average: bool,
  ) -> jnp.ndarray:
    return jnp.zeros([inputs.shape[0], self.output_dim])


class TrainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.train_dir = tempfile.TemporaryDirectory("train_dir").name

    self.data_dir = tempfile.TemporaryDirectory("data_dir").name
    fake_builder = fake_dataset.FakeDataset(data_dir=self.data_dir)
    fake_builder.download_and_prepare()
    self.builder = fake_builder

  def _get_test_dataset(self, config):
    ds, dataset_info = data_utils.get_dataset(
        "train",
        dataset_directory=self.builder.data_dir,
        pipeline=config.train_dataset_config.pipeline,
    )
    return ds, dataset_info

  def _get_test_config(self, config_module=baseline) -> config_dict.ConfigDict:
    """Reduces test config sizes to avoid memory blowouts."""
    config = config_module.get_config()
    config.sample_rate_hz = 11_025
    config.num_train_steps = 1
    config.train_window_size_s = TEST_WINDOW_S
    config.eval_window_size_s = TEST_WINDOW_S
    config.train_config.log_every_steps = 1
    config.train_config.checkpoint_every_steps = 1
    config.eval_config.eval_steps_per_checkpoint = 1
    config = config_utils.parse_config(config, config_globals.get_globals())

    config.train_dataset_config.pipeline = pipeline.Pipeline(
        ops=[
            pipeline.OnlyJaxTypes(),
            pipeline.ConvertBirdTaxonomyLabels(
                source_namespace="ebird2021",
                target_class_list="xenocanto",
                add_taxonomic_labels=True,
            ),
            pipeline.MixAudio(mixin_prob=0.0),
            pipeline.Batch(batch_size=1, split_across_devices=True),
            pipeline.RandomSlice(window_size=TEST_WINDOW_S),
            pipeline.RandomNormalizeAudio(min_gain=0.15, max_gain=0.25),
        ]
    )

    config.eval_dataset_config.pipeline = pipeline.Pipeline(
        ops=[
            pipeline.OnlyJaxTypes(),
            pipeline.MultiHot(),
            pipeline.Batch(batch_size=1, split_across_devices=True),
            pipeline.Slice(
                window_size=TEST_WINDOW_S, start=0.5, names=("audio",)
            ),
            pipeline.NormalizeAudio(target_gain=0.2, names=("audio",)),
        ]
    )

    return config

  def _add_const_model_config(self, config):
    config.init_config.model_config.encoder = ConstantEncoder(output_dim=32)
    return config

  def _add_b0_model_config(self, config):
    config.init_config.model_config.encoder = efficientnet.EfficientNet(
        efficientnet.EfficientNetModel.B0
    )
    return config

  def _add_pcen_melspec_frontend(self, config):
    config.init_config.model_config.frontend = frontend.MelSpectrogram(
        features=32,
        stride=32_000 // 25,
        kernel_size=2_560,
        sample_rate=32_000,
        freq_range=(60, 10_000),
        scaling_config=frontend.PCENScalingConfig(conv_width=256),
    )
    return config

  def test_config_structure(self):
    # Check that the test config and model config have similar structure.
    # This helps ensure that the test configs don't drift too far from the
    # actual configs we use for training.
    raw_config = baseline.get_config()
    parsed_config = config_utils.parse_config(
        raw_config, config_globals.get_globals()
    )
    test_config = self._get_test_config()
    test_config = self._add_pcen_melspec_frontend(test_config)
    test_config = self._add_b0_model_config(test_config)
    print(jax.tree_util.tree_structure(parsed_config.to_dict()))
    print(jax.tree_util.tree_structure(test_config.to_dict()))
    self.assertEqual(
        jax.tree_util.tree_structure(parsed_config.to_dict()),
        jax.tree_util.tree_structure(test_config.to_dict()),
    )

  def test_export_model(self):
    # NOTE: This test might fail when run on a machine that has a GPU but when
    # CUDA is not linked (JAX will detect the GPU so jax2tf will try to create
    # a TF graph on the GPU and fail)
    config = self._get_test_config()
    config = self._add_const_model_config(config)
    config = self._add_pcen_melspec_frontend(config)

    model_bundle, train_state = classifier.initialize_model(
        workdir=self.train_dir, **config.init_config
    )
    train_state = model_bundle.ckpt.restore_or_initialize(train_state)

    classifier.export_tf_model(
        model_bundle,
        train_state,
        self.train_dir,
        config.init_config.input_shape,
        num_train_steps=0,
        eval_sleep_s=0,
    )
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(self.train_dir, "model.tflite"))
    )
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(self.train_dir, "label.csv"))
    )
    with open(os.path.join(self.train_dir, "label.csv")) as f:
      got_class_list = namespace.ClassList.from_csv(f.readlines())
    class_lists = {
        md.key: md.class_list for md in config.init_config.output_head_metadatas
    }
    class_list = class_lists["label"]
    self.assertEqual(class_list.classes, got_class_list.classes)

    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(self.train_dir, "savedmodel/saved_model.pb")
        )
    )

    # Check that saved_model inference doesn't crash.
    # Currently lax.scan (used in the non-convolutional PCEN) fails.
    # See: https://github.com/google/jax/issues/12504
    # The convolutional EMA (using conv_width != 0) works, though.
    reloaded = tf.saved_model.load(os.path.join(self.train_dir, "savedmodel"))
    audio = jnp.zeros([1, config.sample_rate_hz])
    reloaded.infer_tf(audio)

  @parameterized.parameters(
      baseline,
      baseline_attention,
      baseline_mel_conformer,
  )
  def test_init(self, config_module):
    # Ensure that we can initialize the model with the each config.
    config = self._get_test_config(config_module)
    # Check that field reference for num_train_steps propogated appropriately.
    self.assertEqual(
        config.train_config.num_train_steps, config.eval_config.num_train_steps
    )

    model_bundle, train_state = classifier.initialize_model(
        workdir=self.train_dir, **config.init_config
    )
    self.assertIsNotNone(model_bundle)
    self.assertIsNotNone(train_state)

  def test_train_one_step(self):
    config = self._get_test_config()
    config = self._add_const_model_config(config)
    config = self._add_pcen_melspec_frontend(config)
    ds, _ = self._get_test_dataset(config)
    model_bundle, train_state = classifier.initialize_model(
        workdir=self.train_dir, **config.init_config
    )

    classifier.train(
        model_bundle=model_bundle,
        train_state=train_state,
        train_dataset=ds,
        logdir=self.train_dir,
        **config.train_config,
    )
    ckpt = checkpoint.MultihostCheckpoint(self.train_dir)
    self.assertIsNotNone(ckpt.latest_checkpoint)

  def test_eval_one_step(self):
    config = self._get_test_config()
    config = self._add_const_model_config(config)
    config = self._add_pcen_melspec_frontend(config)
    ds, _ = self._get_test_dataset(config)
    model_bundle, train_state = classifier.initialize_model(
        workdir=self.train_dir, **config.init_config
    )
    # Write a checkpoint, or else the eval will hang.
    model_bundle.ckpt.save(train_state)

    config.eval_config.num_train_steps = 0
    classifier.evaluate(
        model_bundle=model_bundle,
        train_state=train_state,
        valid_dataset=ds,
        workdir=self.train_dir,
        eval_sleep_s=0,
        **config.eval_config,
    )
    ckpt = checkpoint.MultihostCheckpoint(self.train_dir)
    self.assertIsNotNone(ckpt.latest_checkpoint)


if __name__ == "__main__":
  absltest.main()
