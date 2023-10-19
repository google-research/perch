# coding=utf-8
# Copyright 2023 The Perch Authors.
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

"""Configuration to run baseline model."""

from chirp import config_utils
from chirp.configs import presets
from ml_collections import config_dict

# Dataset dirs
REEF_DATASET_DIR = ''
BIRD_DATASET_DIR = ''
# We hold out some reef data for eval
HELD_OUT_DATA = '/_/heldout_filenames.csv'

# Training steps
num_train_steps = 10

# Hyperparams used by get_hyper and config.
batch_size = [64]
learning_rate = [0.01]
encoder_architecture = ['b1']
bird_weight = [0.5]

_c = config_utils.callable_config


# Define separate reef and bird pipelines
def get_reef_pipeline(
    config: config_dict.ConfigDict, mixin_prob: float, train_dataset_dir: str
) -> config_dict.ConfigDict:
  """Create the supervised training data pipeline."""
  pipeline = _c(
      'pipeline.Pipeline',
      ops=[
          _c('pipeline.Shuffle', shuffle_buffer_size=512),
          _c('pipeline.Repeat'),
          _c(
              'pipeline.FilterByFeature',
              filtering_df_path=HELD_OUT_DATA,
              complement=True,
          ),
          _c(
              'pipeline.ConvertReefLabels',
              source_namespace='all_reefs',
              target_class_list='all_reefs',
          ),
          # OnlyJaxTypes must come after ConvertReefLabels
          _c('pipeline.OnlyJaxTypes'),
          _c('pipeline.RandomNormalizeAudio', min_gain=0.15, max_gain=0.25),
          # The keys in the dataset do not match the MixAudio default keys. This
          # was giving a batch shape error. So redefine keys here.
          _c(
              'pipeline.MixAudio',
              mixin_prob=mixin_prob,
              pad_names=[
                  'annotation_end',
                  'annotation_start',
                  'segment_end',
                  'segment_start',
                  'recording_id',
                  'segment_id',
              ],
              label_names=['reef_label', 'reef_label_mask'],
          ),
          _c(
              'pipeline.RepeatPadding',
              pad_size=config.get_ref('train_window_size_s'),
              sample_rate=config.get_ref('sample_rate_hz'),
          ),
          # Some unnecesary keys are introduced, we remove to minimise zero
          # tensors which must be added when mixing with birds
          _c(
              'pipeline.RemoveUnwantedKeys',
              unwanted_keys=[
                  'annotation_end',
                  'annotation_start',
                  'audio_mask',
                  'source_audio',
              ],
          ),
      ],
  )
  return pipeline


def get_bird_pipeline(
    config: config_dict.ConfigDict, mixin_prob: float, train_dataset_dir: str
) -> config_dict.ConfigDict:
  """Create the supervised training data pipeline."""
  pipeline = _c(
      'pipeline.Pipeline',
      ops=[
          _c('pipeline.Shuffle', shuffle_buffer_size=512),
          _c('pipeline.Repeat'),
          _c('pipeline.OnlyJaxTypes'),
          _c(
              'pipeline.ConvertBirdTaxonomyLabels',
              source_namespace='ebird2021',
              target_class_list=config.get_ref('target_class_list'),
              add_taxonomic_labels=config.get_ref('add_taxonomic_labels'),
          ),
          _c('pipeline.RandomNormalizeAudio', min_gain=0.15, max_gain=0.25),
          _c(
              'pipeline.RandomSlice',
              window_size=config.get_ref('train_window_size_s'),
          ),
          _c('pipeline.MixAudio', mixin_prob=mixin_prob),
          _c(
              'pipeline.Pad',
              pad_size=config.get_ref('train_window_size_s'),
              add_mask=config.get_ref('pad_mask'),
          ),
          # source_audio is used by RandomNormalizeAudio above. Drop it here to
          # prevent conflict with reef audio, as bird source_audio was longer
          _c(
              'pipeline.RemoveUnwantedKeys',
              unwanted_keys=['source_audio'],
          ),
      ],
  )
  return pipeline


def get_final_pipeline(
    config: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
  """Pipeline used on a unified dataset object made from multiple datasets."""
  pipeline = _c(
      'pipeline.Pipeline',
      ops=[
          _c(
              'pipeline.Batch',
              batch_size=config.get_ref('batch_size'),
              split_across_devices=True,
          ),
      ],
  )
  return pipeline


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  config = presets.get_base_config()

  # Configure the data
  bird_pipeline = get_bird_pipeline(
      config,
      mixin_prob=0.75,
      train_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
  )
  reef_pipeline = get_reef_pipeline(
      config,
      mixin_prob=0.75,
      train_dataset_dir='reefs_benwilliamsgpt:1.3.0',
  )
  config.is_multi_dataset = True
  config.train_dataset_config = config_dict.ConfigDict()
  config.train_dataset_config.pipelines = [bird_pipeline, reef_pipeline]
  config.train_dataset_config.dataset_directories = [
      'bird_taxonomy/slice_peaked:2.1.0',
      'reefs_benwilliamsgpt:1.3.0',
  ]
  config.train_dataset_config.tfds_data_dirs = [
      BIRD_DATASET_DIR,
      REEF_DATASET_DIR,
  ]
  config.bird_weight = 0.5
  config.train_dataset_config.weights = [
      config.get_ref('bird_weight'),
      1.0 - config.get_ref('bird_weight'),
  ]
  config.train_dataset_config.final_pipeline = get_final_pipeline(config)
  config.train_dataset_config.split = 'train'
  config.num_train_steps = num_train_steps
  config.batch_size = 32
  config.sample_rate_hz = 32000
  config.train_window_size_s = 5  # only takes int so stick to 5s for now
  config.eval_window_size_s = 5
  config.add_taxonomic_labels = True

  # Configure the experiment setup
  config.init_config = presets.get_classifier_init_config(config)
  config.init_config.optimizer = _c(
      'optax.adam', learning_rate=config.init_config.get_ref('learning_rate')
  )
  config.init_config.output_head_metadatas = (
      _c(
          'train_utils.OutputHeadMetadata.from_db',
          key='label',
          class_list_name=config.get_ref('target_class_list'),
          weight=1.0,
      ),
      _c(
          'train_utils.OutputHeadMetadata.from_mapping',
          key='genus',
          source_class_list_name=config.get_ref('target_class_list'),
          weight=0.1,
          mapping_name='ebird2021_to_genus',
      ),
      _c(
          'train_utils.OutputHeadMetadata.from_mapping',
          key='family',
          source_class_list_name=config.get_ref('target_class_list'),
          weight=0.1,
          mapping_name='ebird2021_to_family',
      ),
      _c(
          'train_utils.OutputHeadMetadata.from_mapping',
          key='order',
          source_class_list_name=config.get_ref('target_class_list'),
          weight=0.1,
          mapping_name='ebird2021_to_order',
      ),
      _c(
          'train_utils.OutputHeadMetadata.from_db',
          key='reef_label',
          class_list_name='all_reefs',
          weight=1.0,
      ),
  )
  model_config = config_dict.ConfigDict()
  model_config.encoder = _c(
      'efficientnet.EfficientNet',
      model=_c(
          'efficientnet.EfficientNetModel',
          value='b1',
      ),
  )
  model_config.taxonomy_loss_weight = 0.001
  model_config.frontend = presets.get_bio_pcen_melspec_config(config)
  config.init_config.model_config = model_config
  # Configure the training loop
  config.train_config = presets.get_base_train_config(config)
  config.eval_config = presets.get_base_eval_config(config)

  config.export_config = config_dict.ConfigDict()
  config.export_config.input_shape = (
      config.get_ref('eval_window_size_s') * config.get_ref('sample_rate_hz'),
  )
  config.export_config.num_train_steps = config.get_ref('num_train_steps')

  return config


# For hyperparam sweeps.
def get_hyper(hyper):
  """Defines the hyperparameter sweep."""
  return hyper.product([
      hyper.sweep(
          'config.batch_size',
          hyper.discrete(batch_size),
      ),
      hyper.sweep(
          'config.init_config.learning_rate',
          hyper.discrete(learning_rate),
      ),
      hyper.sweep(
          'config.init_config.model_config.encoder.__config.model.__config.value',
          hyper.categorical(encoder_architecture),
      ),
      hyper.sweep('config.bird_weight', hyper.discrete(bird_weight)),
  ])
