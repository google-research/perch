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

"""Classification over embeddings."""

import dataclasses
from typing import Sequence

from chirp.inference import interface
from chirp.inference import tf_examples
from chirp.inference.classify import data_lib
from chirp.models import metrics
import numpy as np
import tensorflow as tf
import tqdm


@dataclasses.dataclass
class ClassifierMetrics:
  top1_accuracy: float
  auc_roc: float
  recall: float
  cmap_value: float
  class_maps: dict[str, float]
  test_logits: dict[str, np.ndarray]


def get_two_layer_model(
    num_hiddens: int, embedding_dim: int, num_classes: int, batch_norm: bool
) -> tf.keras.Model:
  """Create a simple two-layer Keras model."""
  layers = [tf.keras.Input(shape=[embedding_dim])]
  if batch_norm:
    layers.append(tf.keras.layers.BatchNormalization())
  layers += [
      tf.keras.layers.Dense(num_hiddens, activation='relu'),
      tf.keras.layers.Dense(num_classes),
  ]
  model = tf.keras.Sequential(layers)
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
          tf.keras.metrics.AUC(
              curve='ROC', name='auc', from_logits=True, multi_label=True
          ),
      ],
  )

  train_features = merged.data['embeddings'][train_locs]
  train_labels = merged.data['label_hot'][train_locs]

  model.fit(
      train_features,
      train_labels,
      epochs=num_epochs,
      verbose=0,
      batch_size=batch_size,
  )

  # Compute overall metrics to avoid online approximation error in Keras.
  test_features = merged.data['embeddings'][test_locs]
  test_logits = model.predict(test_features, verbose=0, batch_size=8)
  test_labels_hot = merged.data['label_hot'][test_locs]
  test_labels = merged.data['label'][test_locs]

  # Create a dictionary of test logits for each class.
  test_logits_dict = {}
  for k in set(test_labels):
    lbl_locs = np.argwhere(test_labels == k)[:, 0]
    test_logits_dict[k] = test_logits[lbl_locs, k]

  top_logit_idxs = np.argmax(test_logits, axis=1)
  top1acc = np.mean(test_labels == top_logit_idxs)
  # TODO(tomdenton): Implement recall@precision metric.
  recall = -1.0

  cmap_value = metrics.cmap(test_logits, test_labels_hot)['macro']
  auc_roc = metrics.roc_auc(test_logits, test_labels_hot)
  return ClassifierMetrics(
      top1acc,
      auc_roc['macro'],
      recall,
      cmap_value,
      auc_roc['individual'],
      test_logits_dict,
  )


def train_embedding_model(
    model: tf.keras.Model,
    merged: data_lib.MergedDataset,
    train_ratio: float | None,
    train_examples_per_class: int | None,
    num_epochs: int,
    random_seed: int,
    batch_size: int,
    learning_rate: float | None = None,
    exclude_eval_classes: Sequence[int] = (),
) -> ClassifierMetrics:
  """Trains a classification model over embeddings and labels."""
  train_locs, test_locs, _ = merged.create_random_train_test_split(
      train_ratio, train_examples_per_class, random_seed,
      exclude_eval_classes=exclude_eval_classes,
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


def get_inference_dataset(
    embeddings_ds: tf.data.Dataset,
    model: interface.LogitsOutputHead,
):
  """Create a dataset which includes the model's predictions."""

  def classify_batch(batch):
    """Classify a batch of embeddings."""
    emb = batch[tf_examples.EMBEDDING]
    emb_shape = tf.shape(emb)
    flat_emb = tf.reshape(emb, [-1, emb_shape[-1]])
    logits = model(flat_emb)
    logits = tf.reshape(
        logits, [emb_shape[0], emb_shape[1], tf.shape(logits)[-1]]
    )
    # Take the maximum logit over channels.
    logits = tf.reduce_max(logits, axis=-2)
    batch['logits'] = logits
    return batch

  inference_ds = embeddings_ds.map(
      classify_batch, num_parallel_calls=tf.data.AUTOTUNE
  )
  return inference_ds


def write_inference_csv(
    embeddings_ds: tf.data.Dataset,
    model: interface.LogitsOutputHead,
    labels: Sequence[str],
    output_filepath: str,
    embedding_hop_size_s: float,
    threshold: dict[str, float] | None = None,
    exclude_classes: Sequence[str] = ('unknown',),
    include_classes: Sequence[str] = (),
):
  """Write a CSV file of inference results."""
  inference_ds = get_inference_dataset(embeddings_ds, model)

  detection_count = 0
  nondetection_count = 0
  with open(output_filepath, 'w') as f:
    # Write column headers.
    headers = ['filename', 'timestamp_s', 'label', 'logit']
    f.write(', '.join(headers) + '\n')
    for ex in tqdm.tqdm(inference_ds.as_numpy_iterator()):
      for t in range(ex['logits'].shape[0]):
        for i, label in enumerate(labels):
          if label in exclude_classes:
            continue
          if include_classes and label not in include_classes:
            continue
          if threshold is None or ex['logits'][t, i] > threshold[label]:
            offset = ex['timestamp_s'] + t * embedding_hop_size_s
            logit = '{:.2f}'.format(ex['logits'][t, i])
            row = [
                ex['filename'].decode('utf-8'),
                '{:.2f}'.format(offset),
                label,
                logit,
            ]
            f.write(','.join(row) + '\n')
            detection_count += 1
          else:
            nondetection_count += 1
  print('\n\n\n   Detection count: ', detection_count)
  print('NonDetection count: ', nondetection_count)
