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

"""Classification over embeddings."""

from chirp.projects.multicluster import data_lib
import tensorflow as tf


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


def train_embedding_model(
    model: tf.keras.Model,
    merged: data_lib.MergedDataset,
    num_training_examples: int,
    num_epochs: int,
    random_seed: int,
    batch_size: int,
) -> tuple[float, float]:
  """Trains a classification model over embeddings and labels."""
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=[
          tf.keras.metrics.BinaryAccuracy(threshold=0.0),
          tf.keras.metrics.AUC(curve='PR', name='auc'),
      ],
  )

  train_locs, test_locs, _ = merged.create_random_train_test_split(
      num_training_examples, random_seed
  )
  train_ds = merged.create_keras_dataset(train_locs, True, batch_size)
  test_ds = merged.create_keras_dataset(test_locs, False, batch_size)

  model.fit(train_ds, epochs=num_epochs, verbose=0)
  _, acc, auc = model.evaluate(test_ds)
  return acc, auc
