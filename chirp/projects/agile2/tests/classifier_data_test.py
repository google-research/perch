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

"""Tests for classifier data management."""

import shutil
import tempfile

from chirp import path_utils
from chirp.projects.agile2 import classifier_data
from chirp.projects.agile2 import ingest_annotations
from chirp.projects.agile2.tests import test_utils
from chirp.projects.hoplite import interface
from chirp.taxonomy import annotations_fns
import numpy as np

from absl.testing import absltest


class ClassifierDataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def make_db_with_labels(
      self,
      num_embeddings: int,
      unlabeled_prob: float,
      positive_label_prob: float,
      rng: np.random.Generator,
  ) -> interface.GraphSearchDBInterface:
    db = test_utils.make_db(
        self.tempdir,
        'in_mem',
        num_embeddings=num_embeddings,
        rng=rng,
    )
    test_utils.add_random_labels(
        db,
        rng=rng,
        unlabeled_prob=unlabeled_prob,
        positive_label_prob=positive_label_prob,
    )
    return db

  def test_train_test_split_fully_labeled(self):
    num_embeddings = 5_000
    db = self.make_db_with_labels(
        num_embeddings=num_embeddings,
        unlabeled_prob=0.0,
        positive_label_prob=0.5,
        rng=np.random.default_rng(42),
    )
    data_manager = classifier_data.AgileDataManager(
        target_labels=test_utils.CLASS_LABELS,
        db=db,
        train_ratio=0.8,
        min_eval_examples=10,
        batch_size=10,
        weak_negatives_batch_size=10,
        rng=np.random.default_rng(42),
    )
    train_ids, eval_ids = data_manager.get_train_test_split()
    # Because classes can overlap, we can't guarantee that the train and eval
    # sets are exactly 80% and 20%, but we can check that they are close.
    self.assertGreater(len(eval_ids), 0.9 * 0.2 * num_embeddings)
    self.assertLess(len(eval_ids), 1.1 * 0.2 * num_embeddings)
    self.assertLen(train_ids, num_embeddings - len(eval_ids))
    # But we really want to check that the two sets are disjoint.
    self.assertEmpty(np.intersect1d(train_ids, eval_ids))

    # Check that the labeled examples are split correctly.
    for label in test_utils.CLASS_LABELS:
      class_label_ids = db.get_embeddings_by_label(label, label_type=None)
      eval_count = np.intersect1d(class_label_ids, eval_ids).shape[0]
      self.assertGreaterEqual(eval_count, 10)

  def test_train_test_split_partially_labeled(self):
    num_embeddings = 5_000
    unlabeled_prob = 0.5
    db = self.make_db_with_labels(
        num_embeddings=num_embeddings,
        unlabeled_prob=unlabeled_prob,
        positive_label_prob=0.5,
        rng=np.random.default_rng(42),
    )
    data_manager = classifier_data.AgileDataManager(
        target_labels=test_utils.CLASS_LABELS,
        db=db,
        train_ratio=0.8,
        min_eval_examples=10,
        batch_size=10,
        weak_negatives_batch_size=10,
        rng=np.random.default_rng(42),
    )
    train_ids, eval_ids = data_manager.get_train_test_split()
    # Because classes can overlap, we can't guarantee that the train and eval
    # sets are exactly 80% and 20%, but we can check that they are close.
    self.assertGreater(
        len(eval_ids), unlabeled_prob * 0.9 * 0.2 * num_embeddings
    )
    self.assertLess(len(eval_ids), unlabeled_prob * 1.1 * 0.2 * num_embeddings)
    self.assertGreater(
        len(train_ids), unlabeled_prob * 0.9 * 0.8 * num_embeddings
    )
    self.assertLess(len(train_ids), unlabeled_prob * 1.1 * 0.8 * num_embeddings)
    # But we really want to check that the two sets are disjoint.
    self.assertEmpty(np.intersect1d(train_ids, eval_ids))

  def test_partial_classes(self):
    num_embeddings = 5_000
    db = self.make_db_with_labels(
        num_embeddings=num_embeddings,
        # Add a label to every embedding.
        unlabeled_prob=0.0,
        positive_label_prob=0.5,
        rng=np.random.default_rng(42),
    )
    # Only use three labels, which is half the length of the full class list.
    data_manager = classifier_data.AgileDataManager(
        target_labels=test_utils.CLASS_LABELS[:3],
        db=db,
        train_ratio=0.8,
        min_eval_examples=10,
        batch_size=10,
        weak_negatives_batch_size=10,
        rng=np.random.default_rng(42),
    )
    train_ids, eval_ids = data_manager.get_train_test_split()
    self.assertEmpty(np.intersect1d(train_ids, eval_ids))
    for label in test_utils.CLASS_LABELS[:3]:
      class_label_ids = db.get_embeddings_by_label(label, label_type=None)
      eval_count = np.intersect1d(class_label_ids, eval_ids).shape[0]
      self.assertGreaterEqual(eval_count, 10)
    # Show that examples without target labels are not included.
    for label in test_utils.CLASS_LABELS[3:]:
      class_label_ids = db.get_embeddings_by_label(label, label_type=None)
      eval_count = np.intersect1d(class_label_ids, eval_ids).shape[0]
      train_count = np.intersect1d(class_label_ids, train_ids).shape[0]
      self.assertEqual(eval_count, 0)
      self.assertEqual(train_count, 0)

  def test_multihot_labels(self):
    db = test_utils.make_db(
        self.tempdir,
        'in_mem',
        num_embeddings=100,
        rng=np.random.default_rng(42),
    )
    data_manager = classifier_data.AgileDataManager(
        target_labels=test_utils.CLASS_LABELS,
        db=db,
        train_ratio=0.8,
        min_eval_examples=10,
        batch_size=10,
        weak_negatives_batch_size=10,
        rng=np.random.default_rng(42),
    )

    add_label = lambda id, lbl_idx, lbl_type: db.insert_label(
        interface.Label(
            embedding_id=id,
            label=test_utils.CLASS_LABELS[lbl_idx],
            type=lbl_type,
            provenance='test',
        )
    )

    with self.subTest('single_positive_label'):
      add_label(1, 3, interface.LabelType.POSITIVE)
      multihot, mask = data_manager.get_multihot_labels(1)
      np.testing.assert_equal(multihot, (0, 0, 0, 1, 0, 0))
      np.testing.assert_equal(mask, (0, 0, 0, 1, 0, 0))

    with self.subTest('single_negative_label'):
      add_label(2, 3, interface.LabelType.NEGATIVE)
      multihot, mask = data_manager.get_multihot_labels(2)
      np.testing.assert_equal(multihot, (0, 0, 0, 0, 0, 0))
      np.testing.assert_equal(mask, (0, 0, 0, 1, 0, 0))

    with self.subTest('disagreeing_labels'):
      add_label(3, 1, interface.LabelType.POSITIVE)
      add_label(3, 1, interface.LabelType.POSITIVE)
      add_label(3, 1, interface.LabelType.POSITIVE)
      add_label(3, 1, interface.LabelType.NEGATIVE)
      multihot, mask = data_manager.get_multihot_labels(3)
      np.testing.assert_equal(multihot, (0, 0.75, 0, 0, 0, 0))
      np.testing.assert_equal(mask, (0, 1, 0, 0, 0, 0))

    with self.subTest('multiple_labels'):
      add_label(4, 1, interface.LabelType.POSITIVE)
      add_label(4, 1, interface.LabelType.POSITIVE)
      add_label(4, 2, interface.LabelType.POSITIVE)
      add_label(4, 3, interface.LabelType.NEGATIVE)
      add_label(4, 5, interface.LabelType.POSITIVE)
      multihot, mask = data_manager.get_multihot_labels(4)
      np.testing.assert_equal(multihot, (0, 1, 1, 0, 0, 1))
      np.testing.assert_equal(mask, (0, 1, 1, 1, 0, 1))

  def test_ingest_annotations(self):
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
        dataset_name='hawaii',
        annotation_filename='hawaii.csv',
        annotation_load_fn=annotations_fns.load_cornell_annotations,
        window_size_s=5.0,
        provenance='test_dataset',
    )
    inserted_labels = ingestor.ingest_dataset(db)
    self.assertSetEqual(inserted_labels, {'jabwar', 'hawama', 'ercfra'})
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


if __name__ == '__main__':
  absltest.main()
