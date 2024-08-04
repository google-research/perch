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

import os
import tempfile
from etils import epath

from chirp.inference import embed_lib, interface
from chirp.inference.classify import classify
from chirp.inference.classify import data_lib
from chirp.inference.search import bootstrap
from chirp.taxonomy import namespace
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from bootstrap_test import BootstrapTest


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

  def test_write_inference_file(self):
    # copy from test_train_linear_model to get the model
    embedding_dim = 128
    num_classes = 4
    model = classify.get_linear_model(embedding_dim, num_classes)
    
    classes = ['a', 'b', 'c', 'd']
    logits_model = interface.LogitsOutputHead(
        model_path='./test_model',
        logits_key='some_model',
        logits_model=model,
        class_list=namespace.ClassList('classes', classes),
    )
    
    # make a fake embeddings dataset
    filenames = ['file1', 'file2', 'file3']
    bt = BootstrapTest()
    bt.setUp()
    audio_glob = bt.make_wav_files(classes, filenames)
    source_infos = embed_lib.create_source_infos([audio_glob], shard_len_s=5.0)

    embed_dir = os.path.join(bt.tempdir, 'embeddings')
    labeled_dir = os.path.join(bt.tempdir, 'labeled')
    epath.Path(embed_dir).mkdir(parents=True, exist_ok=True)
    epath.Path(labeled_dir).mkdir(parents=True, exist_ok=True)
    
    print(source_infos)
    print(bt.tempdir)

    bt.write_placeholder_embeddings(audio_glob, source_infos, embed_dir)

    bootstrap_config = bootstrap.BootstrapConfig.load_from_embedding_path(
        embeddings_path=embed_dir,
        annotated_path=labeled_dir,
    )
    print('config hash : ', bootstrap_config.embedding_config_hash())

    project_state = bootstrap.BootstrapState(
        config=bootstrap_config,
    )
    
    embeddings_ds = project_state.create_embeddings_dataset()
    
    classify.write_inference_file(
        embeddings_ds=embeddings_ds,
        model=logits_model,
        labels=classes,
        output_filepath='./test_output',
        embedding_hop_size_s=5.0,
        row_size=1,
        format='csv'
    )

if __name__ == '__main__':
  absltest.main()
