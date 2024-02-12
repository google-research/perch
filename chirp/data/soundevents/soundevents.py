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

"""soundevents dataset."""

import dataclasses
import tempfile
from typing import Any, Callable, cast
import warnings

from absl import logging
from chirp import audio_utils
from chirp.data import filter_scrub_utils as fsu
from chirp.data import tfds_features
from chirp.taxonomy import namespace_db
from etils import epath
from jax import numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Sound events dataset from FSD50K Audioset dataset.

FSD50K is an open dataset of human-labeled sound events containing 51,197
Freesound clips unequally distributed in 200 classes drawn from the
AudioSet Ontology. (https://zenodo.org/record/4060432#.Y4PE5-xKjzc)

Freesound Dataset 50k (or FSD50K for short) is an open dataset of human-labeled
sound events containing 51,197 Freesound clips unequally distributed in 200
classes drawn from the AudioSet Ontology

We use a slightly different format than AudioSet for the naming of class labels
in order to avoid potential problems with spaces, commas, etc.
Example: Accelerating_and_revving_and_vroom instead of the original
Accelerating, revving, vroom. You can go back to the original AudioSet naming
using the information provided in vocabulary.csv (class label and mid for the
200 classes of FSD50K) and the AudioSet Ontology specification.

Audioset  consists of an expanding ontology of 632 audio event classes and a
collection of 2,084,320 human-labeled 10-second sound clips drawn from
YouTube videos (https://research.google.com/audioset/index.html)

The AudioSet ontology is a collection of sound events organized in a hierarchy.
https://research.google.com/audioset/ontology/index.html

The AudioSet dataset is a large-scale collection of human-labeled 10-second
sound clips drawn from YouTube videos. To collect all our data we worked with
human annotators who verified the presence of sounds they heard within
YouTube segments. (https://research.google.com/audioset/dataset/index.html)

"""

_CITATION = """
@inproceedings{fonseca2022FSD50K,
  title={{FSD50K}: an open dataset of human-labeled sound events},
  author={Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic
  and Serra, Xavier},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={30},
  pages={829--852},
  year={2022},
  publisher={IEEE}
}

"""

LocalizationFn = Callable[[Any, int, float], jnp.ndarray]


@dataclasses.dataclass
class SoundeventsConfig(tfds.core.BuilderConfig):
  """The config to generate multiple versions of Sound Events from FSD50K."""

  sample_rate_hz: int = 32_000
  resampling_method: str = 'polyphase'
  localization_fn: LocalizationFn | None = None
  interval_length_s: float | None = None
  data_processing_query: fsu.QuerySequence = fsu.QuerySequence(queries=[])
  metadata_processing_query: fsu.QuerySequence = fsu.QuerySequence(queries=[])
  class_list_name: str = 'fsd50k'


class Soundevents(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for soundevents dataset."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'Added a config filter out bird classes',
  }

  BUILDER_CONFIGS = [
      # pylint: disable=unexpected-keyword-arg
      SoundeventsConfig(
          name='fsd50k_full_length',
          localization_fn=None,
          class_list_name='fsd50k',
          description='full length audio sequences processed with ',
      ),
      SoundeventsConfig(
          name='fsd50k_slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          class_list_name='fsd50k',
          description=(
              'Chunked audio sequences processed with '
              'chirp.audio_utils.slice_peaked_audio.'
          ),
      ),
      SoundeventsConfig(
          name='fsd50k_no_bird_slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          class_list_name='fsd50k',
          description=(
              'FSD50K dataset excluding bird classes '
              'chunked audio sequences processed with '
              'chirp.audio_utils.slice_peaked_audio.'
          ),
          data_processing_query=fsu.QuerySequence(
              [fsu.filter_contains_no_class_list('class_code', 'fsd50k_birds')]
          ),
      ),
  ]

  GCS_URL = epath.Path('gs://chirp-public-bucket/soundevents/fsd50k')

  DATASET_CONFIG = {
      'dev': {
          'ground_truth_file': GCS_URL / 'FSD50K.ground_truth/dev.csv',
          'audio_dir': GCS_URL / 'dev_audio',
      },
      'eval': {
          'ground_truth_file': GCS_URL / 'FSD50K.ground_truth/eval.csv',
          'audio_dir': GCS_URL / 'eval_audio',
      },
  }

  def _info(self) -> tfds.core.DatasetInfo:
    db = namespace_db.load_db()
    dataset_class_list = db.class_lists[self.builder_config.class_list_name]
    logging.info(
        'Currently considering a total of %s soundevent.',
        len(dataset_class_list.classes),
    )

    full_length = self.builder_config.localization_fn is None
    audio_feature_shape = [
        None
        if full_length
        else int(
            self.builder_config.sample_rate_hz
            * self.builder_config.interval_length_s
        )
    ]

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'audio': tfds_features.Int16AsFloatTensor(
                shape=audio_feature_shape,
                sample_rate=self.builder_config.sample_rate_hz,
                encoding=tfds.features.Encoding.ZLIB,
            ),
            'recording_id': tfds.features.Scalar(dtype=tf.uint64),
            'segment_id': tfds.features.Scalar(dtype=tf.int64),
            'segment_start': tfds.features.Scalar(dtype=tf.uint64),
            'segment_end': tfds.features.Scalar(dtype=tf.uint64),
            'label': tfds.features.Sequence(
                tfds.features.ClassLabel(names=dataset_class_list.classes)
            ),
            'filename': tfds.features.Text(),
            'class_name': tfds.features.Text(),
        }),
        supervised_keys=('audio', 'label'),
        homepage='https://github.com/google-research/perch',
        citation=_CITATION,
    )

  def _load_dataset(
      self, dl_manager: tfds.download.DownloadManager, dataset_type: str
  ) -> pd.DataFrame:
    """Loading FSD50k train or test dataset from bucket.

    'dataset_type' is eaither dev or eval used to seperate train and test
    dataset in FSD50K dataset. This dowload and process ground truth file
    generate source_infor for dataset

    Args:
      dl_manager: Download Manager
      dataset_type: 'train' or 'eval' dataset type. Corresponding ground truth
        files are slightly different format in FSD50K dataset. Dowloading and
        preparing source_infromation

    Returns:
      source_info:  A dataframe contain our source infromation for each data
      element
    """
    dl_manager._force_checksums_validation = (
        False  # pylint: disable=protected-access
    )
    # get ground truth files for dev set which included dev and val examples
    paths = dl_manager.download_and_extract({
        'dataset_info_dev': (
            self.DATASET_CONFIG['dev']['ground_truth_file']
        ).as_posix(),
        'dataset_info_eval': (
            self.DATASET_CONFIG['eval']['ground_truth_file']
        ).as_posix(),
    })
    source_info = pd.read_csv(paths[f'dataset_info_{dataset_type}'])
    if dataset_type == 'eval':
      source_info.columns = ['fname', 'labels', 'mids']
      source_info['split'] = 'test'
    # get_split = lambda s: s['split']
    get_labels = lambda s: s['labels'].split(',')
    get_class_codes = lambda s: s['mids'].split(',')
    get_filename = lambda s: f"{s['fname']}.wav"
    audio_dir = self.DATASET_CONFIG[dataset_type]['audio_dir']
    source_info['url'] = source_info.apply(
        lambda s: audio_dir / f'{get_filename(s)}', axis=1
    )
    source_info['label'] = source_info.apply(get_labels, axis=1)
    source_info['class_code'] = source_info.apply(get_class_codes, axis=1)
    source_info = source_info.drop(['mids', 'labels'], axis=1)
    # Apply all the processing queries.
    # TODO(haritaoglu) need to test processing queries
    source_info = fsu.apply_sequence(
        source_info, self.builder_config.data_processing_query
    )
    if source_info is pd.Series:
      source_info = source_info.to_frame()
    else:
      assert type(source_info) is pd.DataFrame
      source_info = cast(pd.DataFrame, source_info)
    return source_info

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    # Defined as part of the tfds API for dividing the dataset into splits.
    # https://www.tensorflow.org/datasets/add_dataset#specifying_dataset_splits

    train_source_info = self._load_dataset(dl_manager, 'dev')
    eval_source_info = self._load_dataset(dl_manager, 'eval')
    return {
        'train': self._generate_examples(source_info=train_source_info),
        'test': self._generate_examples(source_info=eval_source_info),
    }

  def _generate_examples(self, source_info: pd.DataFrame):
    beam = tfds.core.lazy_imports.apache_beam
    librosa = tfds.core.lazy_imports.librosa

    def _process_example(row):
      recording_id, source = row

      with tempfile.NamedTemporaryFile(
          mode='w+b', suffix=source['url'].suffix
      ) as f:
        f.write(source['url'].read_bytes())
        # librosa outputs lots of warnings which we can safely ignore when
        # processing all souendevents  files and PySoundFile is unavailable.
        with warnings.catch_warnings():
          warnings.simplefilter('ignore')
          audio, _ = librosa.load(
              f.name,
              sr=self.builder_config.sample_rate_hz,
              res_type=self.builder_config.resampling_method,
          )
          # Resampling can introduce artifacts that push the signal outside the
          # [-1, 1) interval.
          audio = np.clip(audio, -1.0, 1.0 - (1.0 / float(1 << 15)))
      label = source['class_code']
      return source['fname'], {
          'audio': audio,
          'recording_id': recording_id,
          'segment_id': -1,
          'segment_start': 0,
          'segment_end': len(audio),
          'label': label,
          'class_name': source['label'][0],
          'filename': source['url'].as_posix(),
      }

    pipeline = beam.Create(source_info.iterrows()) | beam.Map(_process_example)

    if self.builder_config.localization_fn:
      print('Adding Localization_function')

      def _localize_intervals(args):
        key, example = args
        sample_rate_hz = self.builder_config.sample_rate_hz
        interval_length_s = self.builder_config.interval_length_s
        target_length = int(sample_rate_hz * interval_length_s)

        audio = audio_utils.pad_to_length_if_shorter(
            example['audio'], target_length
        )
        # Pass padded audio to avoid localization_fn having to pad again
        audio_intervals = self.builder_config.localization_fn(
            audio, sample_rate_hz, interval_length_s
        ).tolist()

        if not audio_intervals:
          # If no peaks were found, we take the first segment of the
          # recording to avoid discarding it entirely
          audio_intervals = [(0, target_length)]
        interval_examples = []
        for i, (start, end) in enumerate(audio_intervals):
          interval_examples.append((
              f'{key}_{i}',
              {
                  **example,
                  'audio': audio[start:end],
                  'segment_id': i,
                  'segment_start': start,
                  'segment_end': end,
              },
          ))
        print(f' Interval examples : {interval_examples}')
        return interval_examples

      pipeline = pipeline | beam.FlatMap(_localize_intervals)
    return pipeline
