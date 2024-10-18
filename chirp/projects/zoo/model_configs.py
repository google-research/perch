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

"""Convenience functions for predefined model configs."""

import enum

from chirp.models import frontend
from chirp.projects.zoo import models
from chirp.projects.zoo import taxonomy_model_tf
from chirp.projects.zoo import zoo_interface
from ml_collections import config_dict


class ModelConfigName(enum.StrEnum):
  """Names of known preset configs."""

  BEANS_BASELINE = 'beans_baseline'
  BIRDNET_V2_1 = 'birdnet_V2.1'
  BIRDNET_V2_2 = 'birdnet_V2.2'
  BIRDNET_V2_3 = 'birdnet_V2.3'
  PERCH_8 = 'perch_8'
  SURFPERCH = 'surfperch'
  VGGISH = 'vggish'
  YAMNET = 'yamnet'
  HUMPBACK = 'humpback'
  MULTISPECIES_WHALE = 'multispecies_whale'


MODEL_CLASS_MAP: dict[str, type[zoo_interface.EmbeddingModel]] = {
    'taxonomy_model_tf': taxonomy_model_tf.TaxonomyModelTF,
    'separator_model_tf': models.SeparatorModelTF,
    'birb_separator_model_tf1': models.BirbSepModelTF1,
    'birdnet': models.BirdNet,
    'placeholder_model': models.PlaceholderModel,
    'separate_embed_model': models.SeparateEmbedModel,
    'tfhub_model': models.TFHubModel,
    'google_whale': models.GoogleWhaleModel,
    'handcrafted_features': models.HandcraftedFeaturesModel,
}


def load_model_by_name(
    model_config_name: str | ModelConfigName,
) -> zoo_interface.EmbeddingModel:
  """Loads the embedding model by model name."""
  model_config_name = ModelConfigName(model_config_name)
  model_class_key, _, config = get_preset_model_config(model_config_name)
  model_class = MODEL_CLASS_MAP[model_class_key]
  return model_class.from_config(config)


def get_preset_model_config(preset_name: str | ModelConfigName):
  """Get a config_dict for a known model."""
  model_config = config_dict.ConfigDict()
  preset_name = ModelConfigName(preset_name)

  if preset_name == ModelConfigName.PERCH_8:
    model_key = 'taxonomy_model_tf'
    embedding_dim = 1280
    model_config.window_size_s = 5.0
    model_config.hop_size_s = 5.0
    model_config.sample_rate = 32000
    model_config.tfhub_version = 8
    model_config.model_path = ''
  elif preset_name == ModelConfigName.HUMPBACK:
    model_key = 'google_whale'
    embedding_dim = 2048
    model_config.window_size_s = 3.9124
    model_config.sample_rate = 10000
    model_config.model_url = 'https://tfhub.dev/google/humpback_whale/1'
    model_config.peak_norm = 0.02
  elif preset_name == ModelConfigName.MULTISPECIES_WHALE:
    model_key = 'google_whale'
    embedding_dim = 1280
    model_config.window_size_s = 5.0  # Is this correct?
    model_config.sample_rate = 24000
    model_config.model_url = 'https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2'
    model_config.peak_norm = -1.0
  elif preset_name == ModelConfigName.SURFPERCH:
    model_key = 'taxonomy_model_tf'
    embedding_dim = 1280
    model_config.window_size_s = 5.0
    model_config.hop_size_s = 5.0
    model_config.sample_rate = 32000
    model_config.tfhub_version = 1
    model_config.tfhub_path = taxonomy_model_tf.SURFPERCH_TF_HUB_URL
    model_config.model_path = ''
  elif preset_name.value.startswith('birdnet'):
    model_key = 'birdnet'
    birdnet_version = preset_name.split('_')[-1]
    if birdnet_version not in ('V2.1', 'V2.2', 'V2.3'):
      raise ValueError(f'Birdnet version not supported: {birdnet_version}')
    base_path = 'gs://chirp-public-bucket/models/birdnet'
    if birdnet_version == 'V2.1':
      embedding_dim = 420
      model_path = 'V2.1/BirdNET_GLOBAL_2K_V2.1_Model_FP16.tflite'
    elif birdnet_version == 'V2.2':
      embedding_dim = 320
      model_path = 'V2.2/BirdNET_GLOBAL_3K_V2.2_Model_FP16.tflite'
    elif birdnet_version == 'V2.3':
      embedding_dim = 1024
      model_path = 'V2.3/BirdNET_GLOBAL_3K_V2.3_Model_FP16.tflite'
    else:
      # TODO(tomdenton): Support V2.4.
      raise ValueError(f'Birdnet version not supported: {birdnet_version}')
    model_config.window_size_s = 3.0
    model_config.hop_size_s = 3.0
    model_config.sample_rate = 48000
    model_config.model_path = f'{base_path}/{model_path}'
    # Note: The v2_1 class list is appropriate for Birdnet 2.1, 2.2, and 2.3.
    model_config.class_list_name = 'birdnet_v2_1'
    model_config.num_tflite_threads = 4
  elif preset_name == ModelConfigName.YAMNET:
    model_key = 'tfhub_model'
    embedding_dim = 1024
    model_config.sample_rate = 16000
    model_config.model_url = 'https://tfhub.dev/google/yamnet/1'
    model_config.embedding_index = 1
    model_config.logits_index = 0
  elif preset_name == ModelConfigName.VGGISH:
    model_key = 'tfhub_model'
    embedding_dim = 128
    model_config.sample_rate = 16000
    model_config.model_url = 'https://tfhub.dev/google/vggish/1'
    model_config.embedding_index = -1
    model_config.logits_index = -1
  elif preset_name == ModelConfigName.BEANS_BASELINE:
    model_key = 'handcrafted_features'
    embedding_dim = 80
    model_config.sample_rate = 32_000
    model_config.melspec_config = config_dict.ConfigDict({
        'sample_rate': model_config.sample_rate,
        'features': 160,
        'stride': 320,
        'kernel_size': 640,
        'freq_range': (60.0, model_config.sample_rate / 2.0),
        'scaling_config': frontend.LogScalingConfig(),
    })
    model_config.features_config = config_dict.ConfigDict({
        'compute_mfccs': True,
        'aggregation': 'beans',
    })
    model_config.window_size_s = 1.0
    model_config.hop_size_s = 1.0
  else:
    raise ValueError('Unsupported model preset: %s' % preset_name)
  return model_key, embedding_dim, model_config
