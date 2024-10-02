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

"""Tests for annotation ingestion."""

import os
import shutil
import tempfile

from chirp import path_utils
from chirp.projects.agile2 import embed
from chirp.projects.agile2 import ingest_annotations
from chirp.projects.agile2.tests import test_utils
from chirp.projects.hoplite import interface
from chirp.taxonomy import annotations
from chirp.taxonomy import annotations_fns
from etils import epath
from ml_collections import config_dict
import numpy as np

from absl.testing import absltest


class IngestAnnotationsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def make_annotated_data(self):
    sites = ['site_1', 'site_2', 'site_3']
    filenames = ['foo', 'bar', 'baz']
    test_utils.make_wav_files(self.tempdir, sites, filenames, file_len_s=60.0)
    classes = ['x', 'y', 'z']

    # Make a collection of random annotations.
    annos = []
    for i, s in enumerate(sites):
      for j, f in enumerate(filenames):
        for k, c in enumerate(classes):
          annos.append(
              annotations.TimeWindowAnnotation(
                  filename=f'{s}/{f}_{s}.wav',
                  start_time_s=9 * i + 3 * j + k,
                  end_time_s=9 * i + 3 * j + k + 1,
                  namespace='test',
                  label=[c],
              )
          )

    annos_path = os.path.join(self.tempdir, 'annos.csv')
    annotations.write_annotations_csv(annos_path, annos)
    return annos_path, annos

  def test_embed_and_ingest_annotations(self):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(
        self.tempdir,
        'in_mem',
        num_embeddings=0,
        rng=rng,
    )
    emb_offsets = [175, 185, 275, 230, 235]
    emb_idxes = []
    for offset in emb_offsets:
      emb_idx = db.insert_embedding(
          embedding=rng.normal([db.embedding_dimension()]),
          source=interface.EmbeddingSource(
              dataset_name='hawaii',
              source_id='UHH_001_S01_20161121_150000.flac',
              offsets=np.array([offset]),
          ),
      )
      emb_idxes.append(emb_idx)

    hawaii_annos_path = path_utils.get_absolute_path(
        'projects/agile2/tests/testdata/hawaii.csv'
    )
    ingestor = ingest_annotations.AnnotatedDatasetIngestor(
        base_path=hawaii_annos_path.parent,
        audio_glob='*/*.flac',
        dataset_name='hawaii',
        annotation_filename='hawaii.csv',
        annotation_load_fn=annotations_fns.load_cornell_annotations,
    )
    inserted_labels = ingestor.ingest_dataset(
        db, window_size_s=5.0, provenance='test_dataset'
    )
    self.assertSetEqual(
        set(inserted_labels.keys()), {'jabwar', 'hawama', 'ercfra'}
    )
    # Check that individual labels are correctly applied.
    # The Hawai'i test data CSV contains a total of five annotations.
    # The window at offset 175 should have no labels.
    self.assertEmpty(db.get_labels(emb_idxes[0]))  # offset 175

    def _check_label(want_label_str, got_label):
      self.assertEqual(got_label.label, want_label_str)
      self.assertEqual(got_label.type, interface.LabelType.POSITIVE)
      self.assertEqual(got_label.provenance, 'test_dataset')

    # There are two jabwar annotations for the window at offset 185.
    offset_185_labels = db.get_labels(emb_idxes[1])
    self.assertLen(offset_185_labels, 2)
    _check_label('jabwar', offset_185_labels[0])
    _check_label('jabwar', offset_185_labels[1])

    offset_275_labels = db.get_labels(emb_idxes[2])
    self.assertLen(offset_275_labels, 1)
    _check_label('hawama', offset_275_labels[0])

    self.assertEmpty(db.get_labels(emb_idxes[3]))  # offset 230

    offset_235_labels = db.get_labels(emb_idxes[4])
    self.assertLen(offset_235_labels, 1)
    _check_label('ercfra', offset_235_labels[0])

  def test_ingest_annotations(self):
    annos_path, annos = self.make_annotated_data()
    self.assertLen(annos, 27)

    def _loader_fn(x):
      annos = annotations.read_annotations_csv(x, namespace='somedata')
      return annotations.annotations_to_dataframe(annos)

    ingestor = ingest_annotations.AnnotatedDatasetIngestor(
        base_path=epath.Path(self.tempdir),
        audio_glob='*/*.wav',
        dataset_name='test',
        annotation_filename=annos_path,
        annotation_load_fn=_loader_fn,
    )
    placeholder_model_config = config_dict.ConfigDict()
    placeholder_model_config.embedding_size = 32
    placeholder_model_config.sample_rate = 16000
    model_config = embed.ModelConfig(
        model_key='placeholder_model',
        embedding_dim=32,
        model_config=placeholder_model_config,
    )
    db, ingestor_class_counts = ingest_annotations.embed_annotated_dataset(
        ds_choice=ingestor,
        db_path=os.path.join(self.tempdir),
        db_model_config=model_config,
    )
    self.assertEqual(db.count_embeddings(), 60 * 9)

    ingestor.ingest_dataset(db, window_size_s=1.0)
    class_counts = db.get_class_counts()
    for lbl in ('x', 'y', 'z'):
      self.assertEqual(class_counts[lbl], 9)
      self.assertEqual(ingestor_class_counts[lbl], 9)


if __name__ == '__main__':
  absltest.main()
