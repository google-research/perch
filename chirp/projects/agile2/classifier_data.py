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

"""Tools for processing data for the Agile2 classifier."""

import abc
import dataclasses
import itertools
from typing import Any, Iterator, Sequence

from chirp.projects.hoplite import interface
import numpy as np


@dataclasses.dataclass
class LabeledExample:
  """Example container class for model training and evaluation."""

  idx: np.ndarray
  embedding: np.ndarray
  multihot: np.ndarray
  is_labeled_mask: np.ndarray

  @property
  def is_batched(self) -> bool:
    return len(self.embedding.shape) > 1

  @classmethod
  def create_batched(
      cls, sequence: Sequence['LabeledExample']
  ) -> 'LabeledExample':
    return LabeledExample(
        idx=np.array([s.idx for s in sequence]),
        embedding=np.stack([s.embedding for s in sequence], axis=0),
        multihot=np.stack([s.multihot for s in sequence], axis=0),
        is_labeled_mask=np.stack([s.is_labeled_mask for s in sequence], axis=0),
    )

  def join_batches(self, other: 'LabeledExample') -> 'LabeledExample':
    if not self.is_batched or not other.is_batched:
      raise ValueError('Both examples must be batched.')
    return LabeledExample(
        idx=np.concatenate([self.idx, other.idx]),
        embedding=np.concatenate([self.embedding, other.embedding]),
        multihot=np.concatenate([self.multihot, other.multihot]),
        is_labeled_mask=np.concatenate(
            [self.is_labeled_mask, other.is_labeled_mask]
        ),
    )


@dataclasses.dataclass
class DataManager:
  """Base class for managing data for training and evaluation."""

  target_labels: tuple[str, ...]
  db: interface.GraphSearchDBInterface
  batch_size: int
  rng: np.random.Generator

  def get_train_test_split(self) -> tuple[np.ndarray, np.ndarray]:
    """Create a train/test split over all labels.

    Returns:
      Two numpy arrays contianing train and eval embedding ids, respectively.
    """
    raise NotImplementedError('get_train_test_split is not implemented.')

  def get_multihot_labels(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Create the multihot label for one example."""
    labels = self.db.get_labels(idx)
    lbl_idxes = {label: i for i, label in enumerate(self.target_labels)}
    pos = np.zeros(len(self.target_labels), dtype=np.float32)
    neg = np.zeros(len(self.target_labels), dtype=np.float32)
    for label in labels:
      if label.type == interface.LabelType.POSITIVE:
        pos[lbl_idxes[label.label]] += 1.0
      elif label.type == interface.LabelType.NEGATIVE:
        neg[lbl_idxes[label.label]] += 1.0
    count = pos + neg
    mask = count > 0
    denom = np.maximum(count, 1.0)
    multihot = pos / denom
    return multihot, mask

  def labeled_example_iterator(
      self, ids: np.ndarray, repeat: bool = False
  ) -> Iterator[LabeledExample]:
    """Create an iterator for training a classifier for target_labels.

    Args:
      ids: The embedding IDs to iterate over.
      repeat: If True, repeat the iterator indefinitely.

    Yields:
      LabeledExample objects.
    """
    ids = ids.copy()
    self.rng.shuffle(ids)
    q = 0
    while True:
      x = ids[q]
      x_emb = self.db.get_embedding(x)
      x_multihot, x_is_labeled = self.get_multihot_labels(x)
      yield LabeledExample(x, x_emb, x_multihot, x_is_labeled)
      q += 1
      if q >= len(ids) and repeat:
        q = 0
        self.rng.shuffle(ids)
      elif q >= len(ids):
        break

  def batched_example_iterator(
      self,
      labeled_ids: np.ndarray,
      repeat: bool = False,
      **unused_kwargs,
  ) -> Iterator[LabeledExample]:
    """Labeled training data iterator with weak negatives."""
    example_iterator = self.labeled_example_iterator(labeled_ids, repeat=repeat)
    for ex_batch in batched(example_iterator, self.batch_size):
      yield LabeledExample.create_batched(ex_batch)


@dataclasses.dataclass
class AgileDataManager(DataManager):
  """Collects labeled data for training classifiers.

  Attributes:
    target_labels: The labels to use for training.
    db: The database to pull labeled examples from.
    train_ratio: The ratio of labeled examples to use for training.
    min_eval_examples: The minimum number of labeled examples to use for
      evaluation. Note that this is only enforced for positive examples.
    batch_size: The batch size for training.
    weak_negatives_batch_size: The batch size for weak negatives.
    rng: The random number generator to use.
  """
  train_ratio: float
  min_eval_examples: int
  weak_negatives_batch_size: int

  def get_single_label_train_test_split(
      self, label: str
  ) -> tuple[np.ndarray, np.ndarray]:
    """Create a train/test split for a single label."""
    pos_ids = self.db.get_embeddings_by_label(
        label, interface.LabelType.POSITIVE, None
    )
    neg_ids = self.db.get_embeddings_by_label(
        label, interface.LabelType.NEGATIVE, None
    )
    if not pos_ids.shape[0]:
      print('Warning: No positive examples for label: ', label)
      return np.array([], np.int64), np.array([], np.int64)
    all_ids = np.union1d(pos_ids, neg_ids)
    pos_ids = np.unique(pos_ids)
    neg_ids = np.unique(neg_ids)
    # Create a test/train split.
    # Strategy is to carefully create an eval set, and then define the train
    # set as the complement of the eval set amongst the labeled data.
    self.rng.shuffle(pos_ids)
    self.rng.shuffle(neg_ids)

    eval_ratio = 1.0 - self.train_ratio
    if self.min_eval_examples > pos_ids.shape[0]:
      raise ValueError(
          f'min_eval_examples ({self.min_eval_examples}) is greater than the'
          f' number of positive examples ({pos_ids.shape[0]}) for label'
          f' {label}.'
      )
    n_pos_eval = np.maximum(
        int(eval_ratio * pos_ids.shape[0]), self.min_eval_examples
    )
    pos_eval_ids = pos_ids[:n_pos_eval]
    n_neg_eval = np.maximum(
        int(eval_ratio * neg_ids.shape[0]), self.min_eval_examples
    )
    neg_eval_ids = neg_ids[:n_neg_eval]

    eval_ids = np.concatenate([pos_eval_ids, neg_eval_ids], axis=0)
    # De-dupe the eval_ids, in case there are multiple Labels for the embedding.
    eval_ids = np.unique(eval_ids)
    train_ids = np.setdiff1d(all_ids, eval_ids)
    return train_ids, eval_ids

  def get_train_test_split(self) -> tuple[np.ndarray, np.ndarray]:
    """Create a train/test split over all labels.

    Returns:
      Two numpy arrays contianing train and eval embedding ids, respectively.
    """
    train_ids, eval_ids = [], []
    for label in self.target_labels:
      lbl_train, lbl_eval = self.get_single_label_train_test_split(label)
      train_ids.append(lbl_train)
      eval_ids.append(lbl_eval)
    # We need to be careful not to allow labeled examples from any eval sets
    # to leak into the training set, which is a danger for multilabel examples.
    # Take the union of all eval sets, and use the complement as the train set.
    all_eval = np.concatenate(eval_ids, axis=0)
    all_eval = np.unique(all_eval)
    all_ids = np.concatenate(train_ids + eval_ids, axis=0)
    all_ids = np.unique(all_ids)
    all_train = np.setdiff1d(all_ids, all_eval)
    return all_train, all_eval

  def batched_example_iterator(
      self,
      labeled_ids: np.ndarray,
      repeat: bool = False,
      add_weak_negatives: bool = False,
  ) -> Iterator[LabeledExample]:
    """Labeled training data iterator with weak negatives."""
    example_iterator = self.labeled_example_iterator(labeled_ids, repeat=repeat)
    example_iterator = batched(example_iterator, self.batch_size)
    if not add_weak_negatives:
      for ex_batch in example_iterator:
        yield LabeledExample.create_batched(ex_batch)
      return

    weak_ids = np.setdiff1d(self.db.get_embedding_ids(), labeled_ids)
    weak_iterator = self.labeled_example_iterator(weak_ids, repeat=True)
    weak_iterator = batched(weak_iterator, self.weak_negatives_batch_size)

    for ex_batch, weak_ex_batch in zip(example_iterator, weak_iterator):
      # Join the two batches.
      ex_batch = LabeledExample.create_batched(ex_batch)
      weak_ex_batch = LabeledExample.create_batched(weak_ex_batch)
      if add_weak_negatives:
        yield ex_batch.join_batches(weak_ex_batch)
      else:
        yield ex_batch


@dataclasses.dataclass
class FullyAnnotatedDataManager(DataManager):
  """A DataManager for fully-annotated datasets."""

  train_examples_per_class: int
  min_eval_examples: int
  add_unlabeled_train_examples: bool

  def get_train_test_split(self) -> tuple[np.ndarray, np.ndarray]:
    """Create a train/test split over the fully-annotated dataset."""
    pos_id_sets = {}
    eval_id_sets = {}
    for label in self.target_labels:
      pos_id_sets[label] = self.db.get_embeddings_by_label(
          label, interface.LabelType.POSITIVE, None
      )
      self.rng.shuffle(pos_id_sets[label])
      eval_id_sets[label] = pos_id_sets[label][: self.min_eval_examples]
    all_eval_ids = np.concatenate(tuple(eval_id_sets.values()), axis=0)

    # Now produce train sets of the desired size,
    # avoiding the selected eval examples.
    train_id_sets = {}
    for label in self.target_labels:
      pos_set = np.setdiff1d(pos_id_sets[label], all_eval_ids)
      train_id_sets[label] = pos_set[: self.train_examples_per_class]
    if self.add_unlabeled_train_examples:
      unlabeled_ids = np.setdiff1d(
          self.db.get_embedding_ids(),
          np.concatenate(tuple(pos_id_sets.values()), axis=0),
      )
      np.setdiff1d(unlabeled_ids, all_eval_ids)
      train_id_sets['UNLABELED'] = unlabeled_ids[
          : self.train_examples_per_class
      ]

    # The final eval set is the complement of all selected training id's.
    all_train_ids = np.concatenate(tuple(train_id_sets.values()), axis=0)
    eval_ids = np.setdiff1d(self.db.get_embedding_ids(), all_train_ids)
    return all_train_ids, eval_ids


def batched(iterable: Iterator[Any], n: int) -> Iterator[Any]:
  # TODO(tomdenton): Use itertools.batched in Python 3.12+
  # batched('ABCDEFG', 3) → ABC DEF G
  if n < 1:
    raise ValueError('n must be at least one')
  iterator = iter(iterable)
  while batch := tuple(itertools.islice(iterator, n)):
    yield batch
