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

"""Tests for Soundscapes library and specific dataset functionality."""

import os
import tempfile

from absl import logging
from chirp import path_utils
from chirp.data.soundscapes import dataset_fns
from chirp.data.soundscapes import soundscapes
from chirp.data.soundscapes import soundscapes_lib
from chirp.taxonomy import namespace_db
from etils import epath
import librosa
import tensorflow_datasets as tfds

from absl.testing import absltest
from absl.testing import parameterized

BUILDER_CONFIGS = soundscapes.Soundscapes.BUILDER_CONFIGS

SUPERVISED_CONFIGS = [b for b in BUILDER_CONFIGS if b.supervised]
UNSUPERVISED_CONFIGS = [b for b in BUILDER_CONFIGS if not b.supervised]


class SoundscapesLibTest(parameterized.TestCase):

  def _make_audio(self, audio_path, filename, extension, all_audio_filepaths):
    """Creates a test audio file."""
    if not filename.endswith(extension):
      filename = '.'.join([filename, extension])
    audio_filepath = audio_path / filename
    if not audio_filepath.parent.exists():
      audio_filepath.parent.mkdir(parents=True)
    if audio_filepath in all_audio_filepaths:
      return
    tfds.core.lazy_imports.pydub.AudioSegment.silent(duration=100000).export(
        audio_filepath, format=extension
    )
    logging.info('created audio file : %s', audio_filepath.as_posix())
    all_audio_filepaths.append(audio_filepath)
    return audio_filepath

  def setUp(self):
    super(SoundscapesLibTest, self).setUp()
    self.data_dir = tempfile.TemporaryDirectory('data_dir').name
    os.mkdir(self.data_dir)
    # We use the 'caples.csv' only to get the parent directory's path.
    self.testdata_dir = path_utils.get_absolute_epath(
        'tests/testdata/caples.csv'
    ).parent

  def test_load_caples_annotations(self):
    annos_csv_path = path_utils.get_absolute_epath('tests/testdata/caples.csv')
    annos = dataset_fns.load_caples_annotations(annos_csv_path)
    # There are six lines in the example file, but one contains a 'comros'
    # annotation which should be dropped.
    self.assertLen(annos, 5)
    expected_labels = ['gockin', 'dusfly', 'dusfly', 'dusfly', 'yerwar']
    for expected_label, (_, anno) in zip(expected_labels, annos.iterrows()):
      # Caples annotations only contain the filename's stem.
      self.assertLen(anno.filename.split('.'), 1)
      self.assertEqual(anno.namespace, 'ebird2021')
      self.assertEqual(anno.label, [expected_label])

  def test_load_hawaii_annotations(self):
    # Combine the Hawaii 'raw' annotations into a single csv.
    csv_path = epath.Path(self.data_dir) / 'hawaii.csv'
    dataset_fns.combine_hawaii_annotations(self.testdata_dir, csv_path)

    # Then read from the combined csv.
    annos = dataset_fns.load_hawaii_annotations(csv_path)
    # Check that only six annotations are kept, dropping the 'Spectrogram' views
    # which are redundant.
    self.assertLen(annos, 7)
    expected_labels = ['iiwi'] * 4 + ['omao', 'iiwi', 'unknown']
    for expected_label, (_, anno) in zip(expected_labels, annos.iterrows()):
      self.assertEqual(anno.filename, 'hawaii/hawaii_example.wav')
      self.assertEqual(anno.namespace, 'ebird2021')
      self.assertEqual(anno.label, [expected_label])

  def test_load_ssw_annotations(self):
    annos_csv_path = path_utils.get_absolute_epath('tests/testdata/ssw.csv')
    annos = dataset_fns.load_ssw_annotations(annos_csv_path)
    self.assertLen(annos, 4)
    expected_labels = ['cangoo', 'blujay', 'rewbla', 'cangoo']
    for expected_label, (_, anno) in zip(expected_labels, annos.iterrows()):
      self.assertTrue(anno.filename.endswith('.flac'))
      self.assertLen(anno.filename.split('.'), 2)
      self.assertEqual(anno.namespace, 'ebird2021')
      self.assertEqual(anno.label, [expected_label])

  def test_load_birdclef_annotations(self):
    annos_csv_path = path_utils.get_absolute_epath(
        'tests/testdata/birdclef2019_colombia.csv'
    )
    annos = dataset_fns.load_birdclef_annotations(annos_csv_path)
    self.assertLen(annos, 7)
    # This includes an auto-conversion of rufant1 to rufant7.
    expected_labels = [
        'rufant7',
        'kebtou1',
        'stbwre2',
        'stbwre2',
        'yeofly1',
        'bubwre1',
        'fepowl',
    ]
    for expected_label, (_, anno) in zip(expected_labels, annos.iterrows()):
      self.assertTrue(anno.filename.endswith('.wav'))
      self.assertLen(anno.filename.split('.'), 2)
      self.assertEqual(anno.namespace, 'ebird2021')
      self.assertEqual(anno.label, [expected_label])

  def test_load_sierras_kahl_annotations(self):
    annos_csv_path = path_utils.get_absolute_epath(
        'tests/testdata/sierras_kahl.csv'
    )
    annos = dataset_fns.load_sierras_kahl_annotations(annos_csv_path)
    self.assertLen(annos, 4)
    expected_labels = [
        'amerob',
        'amerob',
        'herthr',
        'herthr',
    ]
    for expected_label, (_, anno) in zip(expected_labels, annos.iterrows()):
      self.assertTrue(anno.filename.endswith('.flac'))
      self.assertLen(anno.filename.split('.'), 2)
      self.assertEqual(anno.namespace, 'ebird2021')
      self.assertEqual(anno.label, [expected_label])

  def test_load_powdermill_annotations(self):
    # Combine the Hawaii 'raw' annotations into a single csv.
    combined_csv_path = epath.Path(self.data_dir) / 'powdermill.csv'
    dataset_fns.combine_powdermill_annotations(
        self.testdata_dir / 'powdermill', combined_csv_path
    )

    annos_csv_path = path_utils.get_absolute_epath(
        'tests/testdata/powdermill.csv'
    )
    for csv_path in [combined_csv_path, annos_csv_path]:
      annos = dataset_fns.load_powdermill_annotations(csv_path)
      self.assertLen(annos, 5)
      expected_labels = [
          'norcar',
          'woothr',
          'eastow',
          'eastow',
          'eastow',
      ]
      for expected_label, (_, anno) in zip(expected_labels, annos.iterrows()):
        self.assertTrue(anno.filename.endswith('.wav'))
        self.assertLen(anno.filename.split('.'), 2)
        # Check that we got the nested filepath.
        self.assertEqual(
            anno.filename, 'Recording_1/Recording_1_Segment_05.wav'
        )
        self.assertEqual(anno.namespace, 'ebird2021')
        self.assertEqual(anno.label, [expected_label])

  def test_load_peru_annotations(self):
    annos_csv_path = path_utils.get_absolute_epath('tests/testdata/peru.csv')
    annos = dataset_fns.load_sierras_kahl_annotations(annos_csv_path)
    self.assertLen(annos, 5)
    expected_labels = [
        'blfant1',
        'grasal3',
        'greant1',
        'butwoo1',
        'unknown',  # Mapped from '????'
    ]
    for expected_label, (_, anno) in zip(expected_labels, annos.iterrows()):
      self.assertEqual(anno.filename, 'PER_001_S01_20190116_100007Z.flac')
      self.assertLen(anno.filename.split('.'), 2)
      self.assertEqual(anno.namespace, 'ebird2021')
      self.assertEqual(anno.label, [expected_label])

  def test_load_birdclef_metadata(self):
    md_features = dataset_fns.birdclef_metadata_features()
    metadata = dataset_fns.load_birdclef_metadata(
        self.testdata_dir, md_features
    )
    # Two Colombia metadata files and one SSW file.
    self.assertLen(metadata, 3)

  def test_combine_annotations_with_metadata(self):
    # Currently only birdclef data has metadata to combine. So test that.
    md_features = dataset_fns.birdclef_metadata_features()
    annos_csv_path = path_utils.get_absolute_epath(
        'tests/testdata/birdclef2019_colombia.csv'
    )
    annos = dataset_fns.load_birdclef_annotations(annos_csv_path)
    self.assertLen(annos, 7)

    combined_segments = soundscapes_lib.combine_annotations_with_metadata(
        annos,
        self.testdata_dir,
        md_features,
        dataset_fns.load_birdclef_metadata,
    )
    self.assertLen(combined_segments, 7)
    for feature in md_features.values():
      self.assertIn(feature.target_key, combined_segments.columns.values)
      self.assertNotIn(feature.source_key, combined_segments.columns.values)

      # Check that encoding works.
      for value in combined_segments[feature.target_key].values:
        encoded = feature.feature_type.encode_example(value)
        # These should all be scalar values.
        self.assertEmpty(encoded.shape)

  @parameterized.named_parameters(
      dict(testcase_name='_' + bc.name, builder_config=bc)
      for bc in SUPERVISED_CONFIGS
  )
  def test_create_annotated_segments_df(self, builder_config):
    filename = (
        builder_config.annotation_filename or f'{builder_config.name}.csv'
    )
    annotations_path = self.testdata_dir / filename
    annos = builder_config.annotation_load_fn(annotations_path)
    if not builder_config.supervised:
      raise ValueError('Running a supervised test on an unsupervised config.')

    # Create some audio files.
    audio_path = epath.Path(self.data_dir) / builder_config.name / 'audio'
    all_audio_filepaths = []
    for _, anno in annos.iterrows():
      if anno.filename.endswith('.wav'):
        self._make_audio(audio_path, anno.filename, 'wav', all_audio_filepaths)
      elif anno.filename.endswith('.flac'):
        self._make_audio(audio_path, anno.filename, 'flac', all_audio_filepaths)
      else:
        # Probably just a stem; make a wav file.
        self._make_audio(audio_path, anno.filename, 'wav', all_audio_filepaths)

    # Finally, check that the lights are on.
    segments = soundscapes_lib.create_segments_df(
        all_audio_filepaths=all_audio_filepaths,
        annotations_df=annos,
        supervised=builder_config.supervised,
        metadata_load_fn=builder_config.metadata_load_fn,
        metadata_dir=self.testdata_dir,
        metadata_fields=builder_config.metadata_fields,
    )
    self.assertLen(segments, len(annos))

  def test_get_full_length_annotations(self):
    builder_config = [
        c for c in BUILDER_CONFIGS if c.name == 'caples_full_length'
    ][0]
    filename = (
        builder_config.annotation_filename or f'{builder_config.name}.csv'
    )
    annotations_path = self.testdata_dir / filename
    annos = builder_config.annotation_load_fn(annotations_path)
    audio_path = epath.Path(self.data_dir) / builder_config.name / 'audio'
    all_audio_filepaths = []
    for _, anno in annos.iterrows():
      self._make_audio(audio_path, anno.filename, 'wav', all_audio_filepaths)
    segments = soundscapes_lib.create_segments_df(
        all_audio_filepaths=all_audio_filepaths,
        annotations_df=annos,
        supervised=builder_config.supervised,
        metadata_load_fn=builder_config.metadata_load_fn,
        metadata_dir=self.testdata_dir,
        metadata_fields=builder_config.metadata_fields,
    )
    # The Caples testdata contains only a single file, so no need to subselect.
    self.assertLen(all_audio_filepaths, 1)
    audio, _ = librosa.load(all_audio_filepaths[0], sr=32000)
    db = namespace_db.load_db()
    annotations = soundscapes_lib.get_full_length_annotations(
        audio, segments, db.class_lists['caples'], 32000, unknown_guard=True
    )
    # Check that unknown guard annotations exist.
    self.assertLen(annotations, 7)
    self.assertEqual(annotations['label'][0], 'unknown')
    self.assertEqual(annotations['label'][6], 'unknown')


if __name__ == '__main__':
  absltest.main()
