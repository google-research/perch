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

"""Base configuration for bio-acoustic SFDA experiments."""
from chirp import config_utils
from chirp.projects.sfda import adapt
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  sample_rate_hz = config_dict.FieldReference(32_000)
  target_class_list = config_dict.FieldReference("xenocanto")
  namespace = config_dict.FieldReference("ebird2021")
  add_taxonomic_labels = config_dict.FieldReference(True)

  config = config_dict.ConfigDict()
  config.modality = adapt.Modality.AUDIO
  config.multi_label = True
  config.eval_every = 1  # in epochs

  config.sample_rate_hz = sample_rate_hz
  tfds_data_dir = config_dict.FieldReference("")

  # Configure the data
  batch_size = config_dict.FieldReference(64)
  window_size_s = config_dict.FieldReference(5)

  seed = config_dict.FieldReference(0)

  config.tfds_data_dir = tfds_data_dir
  config.batch_size = batch_size
  config.target_class_list = target_class_list

  adaptation_data_config = config_dict.ConfigDict()
  adaptation_data_config.pipeline = _c(
      "pipeline.Pipeline",
      ops=[
          _c("pipeline.ConvertBirdTaxonomyLabels",
             source_namespace=namespace,
             target_class_list=target_class_list,
             add_taxonomic_labels=add_taxonomic_labels),
          _c("pipeline.Shuffle", shuffle_buffer_size=512, seed=seed),
          _c("sfda_pipeline.Batch",
             batch_size=batch_size,
             split_across_devices=True),
          _c("pipeline.NormalizeAudio", target_gain=0.2),
      ])
  adaptation_data_config.split = "[(0, 75)]"
  adaptation_data_config.tfds_data_dir = tfds_data_dir
  adaptation_data_config.dataset_directory = "soundscapes/high_sierras:1.0.1"
  config.adaptation_data_config = adaptation_data_config

  eval_data_config = config_dict.ConfigDict()
  eval_data_config.pipeline = _c(
      "pipeline.Pipeline",
      ops=[
          _c("pipeline.ConvertBirdTaxonomyLabels",
             source_namespace=namespace,
             target_class_list=target_class_list,
             add_taxonomic_labels=add_taxonomic_labels),
          _c(
              "sfda_pipeline.Batch",
              batch_size=batch_size,
              split_across_devices=True,
          ),
          _c("pipeline.NormalizeAudio", target_gain=0.2),
      ])
  eval_data_config.split = "[(75, 100)]"
  eval_data_config.tfds_data_dir = tfds_data_dir
  eval_data_config.dataset_directory = "soundscapes/high_sierras:1.0.1"

  config.eval_data_config = eval_data_config

  # Configure the experiment setup
  init_config = config_dict.ConfigDict()
  init_config.rng_seed = seed
  init_config.target_class_list = target_class_list
  init_config.input_shape = ((window_size_s * sample_rate_hz).get(),)
  init_config.pretrained_model = True

  # Configure model
  model_config = config_dict.ConfigDict()
  model_config.encoder = _c(
      "efficientnet.EfficientNet",
      model=_c("efficientnet.EfficientNetModel", value="b1"),
  )  # remove any dropout from the model
  model_config.taxonomy_loss_weight = 0.25
  model_config.frontend = _c(
      "frontend.MorletWaveletTransform",
      features=160,
      stride=sample_rate_hz // 100,
      kernel_size=2_048,  # ~0.025 * 32,000
      sample_rate=sample_rate_hz,
      freq_range=(60, 10_000),
      scaling_config=_c("frontend.PCENScalingConfig", conv_width=0),
  )
  init_config.pretrained_ckpt_dir = ""

  config.model_config = model_config
  config.init_config = init_config
  config.eval_mca_every = -1
  return config
