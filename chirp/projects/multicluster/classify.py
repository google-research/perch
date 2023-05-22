# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""Classification over embeddings."""

import dataclasses
from typing import Sequence

from chirp.models import metrics
from chirp.projects.multicluster import data_lib
import tensorflow as tf


@dataclasses.dataclass
class ClassifierMetrics:
  top1_accuracy: float
  auc_roc: float
  recall: float
  cmap_value: float
  class_maps: dict[str, float]


def get_two_layer_model(
    num_hiddens: int, embedding_dim: int, num_classes: int
) -> tf.keras.Model:
  """Create a simple two-layer Keras model."""
  model = tf.keras.Sequential([
      tf.keras.Input(shape=[embedding_dim]),
      tf.keras.layers.Dense(num_hiddens, activation='relu'),
      tf.keras.layers.Dense(num_classes),
  ])
  return model


def get_linear_model(embedding_dim: int, num_classes: int) -> tf.keras.Model:
  """Create a simple linear Keras model."""
  model = tf.keras.Sequential([
      tf.keras.Input(shape=[embedding_dim]),
      tf.keras.layers.Dense(num_classes),
  ])
  return model


def train_from_locs(
    model: tf.keras.Model,
    merged: data_lib.MergedDataset,
    train_locs: Sequence[int],
    test_locs: Sequence[int],
    num_epochs: int,
    batch_size: int,
    learning_rate: float | None = None,
    use_bce_loss: bool = True,
) -> ClassifierMetrics:
  """Trains a classification model over embeddings and labels."""
  if use_bce_loss:
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  else:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=loss,
      metrics=[
          tf.keras.metrics.Precision(top_k=1, name='top1prec'),
          tf.keras.metrics.AUC(curve='ROC', name='auc', from_logits=True),
          tf.keras.metrics.RecallAtPrecision(0.9, name='recall0.9'),
      ],
  )

  train_ds = merged.create_keras_dataset(train_locs, True, batch_size)
  test_ds = merged.create_keras_dataset(test_locs, False, batch_size)

  model.fit(train_ds, epochs=num_epochs, verbose=0)
  _, acc, auc_roc, recall = model.evaluate(test_ds, verbose=1)

  # Manually compute per-class mAP and CmAP scores.
  test_logits = model.predict(test_ds, verbose=0, batch_size=8)
  test_labels = merged.data['label_hot'][test_locs]
  cmap_value = metrics.cmap(test_logits, test_labels)['macro']
  return ClassifierMetrics(acc, auc_roc, recall, cmap_value, {})


def train_embedding_model(
    model: tf.keras.Model,
    merged: data_lib.MergedDataset,
    num_training_examples: int,
    num_epochs: int,
    random_seed: int,
    batch_size: int,
    learning_rate: float | None = None,
) -> ClassifierMetrics:
  """Trains a classification model over embeddings and labels."""
  train_locs, test_locs, _ = merged.create_random_train_test_split(
      num_training_examples, random_seed
  )
  test_metrics = train_from_locs(
      model=model,
      merged=merged,
      train_locs=train_locs,
      test_locs=test_locs,
      num_epochs=num_epochs,
      batch_size=batch_size,
      learning_rate=learning_rate,
  )
  return test_metrics
