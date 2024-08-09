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

"""Test small-model classification."""

import tempfile

from chirp.inference import interface
from chirp.inference.classify import classify
from chirp.inference.classify import data_lib
from chirp.taxonomy import namespace
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class ClassifyTest(parameterized.TestCase):

  def make_merged_dataset(
      self,
      num_points: int,
      rng: np.random.RandomState,
      num_classes: int = 4,
      embedding_dim: int = 16,
  ):
    """Create a MergedDataset with random data."""
    # Merged dataset's data dict contains keys:
    # ['embeddings', 'filename', 'label', 'label_str', 'label_hot']
    data = {}
    data['embeddings'] = np.float32(
        rng.normal(size=(num_points, embedding_dim))
    )
    data['label'] = rng.integers(0, num_classes, size=num_points)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    data['label_str'] = np.array(list(letters[i] for i in data['label']))
    data['label_hot'] = np.zeros((num_points, num_classes), dtype=np.float32)
    for i, label in enumerate(data['label']):
      data['label_hot'][i, label] = 1.0
    return data_lib.MergedDataset(
        data=data,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        labels=letters[:num_classes],
    )

  def test_train_linear_model(self):
    embedding_dim = 16
    num_classes = 4
    num_points = 100
    model = classify.get_linear_model(embedding_dim, num_classes)
    rng = np.random.default_rng(42)
    merged = self.make_merged_dataset(
        num_points=num_points,
        rng=rng,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
    )
    unused_metrics = classify.train_embedding_model(
        model,
        merged,
        train_ratio=0.9,
        train_examples_per_class=None,
        num_epochs=3,
        random_seed=42,
        batch_size=16,
        learning_rate=0.01,
    )
    query = rng.normal(size=(num_points, embedding_dim)).astype(np.float32)

    logits = model(query)

    # Save and restore the model.
    class_names = ['a', 'b', 'c', 'd']
    with tempfile.TemporaryDirectory() as logits_model_dir:
      logits_model = interface.LogitsOutputHead(
          model_path=logits_model_dir,
          logits_key='some_model',
          logits_model=model,
          class_list=namespace.ClassList('classes', class_names),
      )
      logits_model.save_model(logits_model_dir, '')
      restored_model = interface.LogitsOutputHead.from_config_file(
          logits_model_dir
      )
      restored_logits = restored_model(query)
    error = np.abs(restored_logits - logits).sum()
    self.assertEqual(error, 0)


if __name__ == '__main__':
  absltest.main()
