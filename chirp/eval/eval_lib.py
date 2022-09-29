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

"""Utility functions for evaluation."""

import dataclasses
import functools
from typing import Callable, Dict, Generator, Sequence, Tuple

from absl import logging
from chirp.data import pipeline
import jax
import ml_collections
import numpy as np
import pandas as pd
import tensorflow as tf

ConfigDict = ml_collections.ConfigDict

MaskFunction = Callable[[pd.DataFrame], pd.Series]
ClasswiseMaskFunction = Callable[[pd.DataFrame, str], pd.Series]
ClasswiseEvalSetGenerator = Generator[Tuple[str, pd.DataFrame, pd.DataFrame],
                                      None, None]
EvalSetGenerator = Generator[Tuple[str, ClasswiseEvalSetGenerator], None, None]


@dataclasses.dataclass
class EvalSetSpecification:
  """A specification for an eval set.

  Attributes:
    class_names: Class names over which to perform the evaluation.
    search_corpus_global_mask_fn: Function mapping the embeddings dataframe to a
      boolean mask over its rows. Used to represent global properties like
      `df['dataset_name'] == 'birdclef_colombia'`. Computed once and combined
      with `search_corpus_classwise_mask_fn` for every class in `class_names` to
      perform boolean indexing on the embeddings dataframe and form the search
      corpus.
    search_corpus_classwise_mask_fn: Function mapping the embeddings dataframe
      and a class name to a boolean mask over its rows. Used to represent
      classwise properties like `~df['bg_labels'].str.contains(class_name)`.
      Combined with `search_corpus_global_mask_fn` for every class in
      `class_names` to perform boolean indexing on the embeddings dataframe and
      form the search corpus.
    class_representative_global_mask_fn: Function mapping the embeddings
      dataframe to a boolean mask over its rows. Used to represent global
      properties like `df['dataset_name'] == 'xc_downstream'`. Computed once and
      combined with `class_representative_corpus_classwise_mask_fn` for every
      class in `class_names` to perform boolean indexing on the embeddings
      dataframe and form the collection of class representatives.
    class_representative_classwise_mask_fn: Function mapping the embeddings
      dataframe and a class name to a boolean mask over its rows. Used to
      represent classwise properties like
      `df['label'].str.contains(class_name)`. Combined with
      `class_representative_corpus_global_mask_fn` for every class in
      `class_names` to perform boolean indexing on the embeddings dataframe and
      form the collection of class representatives.
    num_representatives_per_class: Number of class representatives to sample. If
      the pool of potential representatives is larger, it's downsampled
      uniformly at random to the correct size.
  """
  class_names: Sequence[str]
  search_corpus_global_mask_fn: MaskFunction
  search_corpus_classwise_mask_fn: ClasswiseMaskFunction
  class_representative_global_mask_fn: MaskFunction
  class_representative_classwise_mask_fn: ClasswiseMaskFunction
  num_representatives_per_class: int


def _load_eval_dataset(dataset_config: ConfigDict) -> tf.data.Dataset:
  """Loads an evaluation dataset from its corresponding configuration dict."""
  return pipeline.get_dataset(
      split=dataset_config.split,
      is_train=False,
      dataset_directory=dataset_config.tfds_name,
      tfds_data_dir=dataset_config.tfds_data_dir,
      tf_data_service_address=None,
      pipeline=dataset_config.pipeline,
  )[0]


def load_eval_datasets(config: ConfigDict) -> Dict[str, tf.data.Dataset]:
  """Loads all evaluation datasets for a given evaluation configuration dict.

  Args:
    config: the evaluation configuration dict.

  Returns:
    A dict mapping dataset names to evaluation datasets.
  """
  return {
      dataset_name: _load_eval_dataset(dataset_config)
      for dataset_name, dataset_config in config.dataset_configs.items()
  }


def get_embeddings(
    dataset: tf.data.Dataset,
    model_callback: Callable[[np.ndarray], np.ndarray]) -> tf.data.Dataset:
  """Embeds the audio slice in each tf.Example across the input dataset.

  Args:
    dataset: a TF Dataset composed of tf.Examples.
    model_callback: a Callable that takes a NumPy array and produces an embedded
      NumPy array.

  Returns:
    An updated TF Dataset with a new 'embedding' feature and deleted 'audio'
    feature.
  """

  def _map_func(example):
    example['embedding'] = tf.py_function(
        func=model_callback, inp=[example['audio']], Tout=tf.float32)
    del example['audio']
    return example

  # Use the 'audio' feature to produce a model embedding; delete the old 'audio'
  # feature.
  return dataset.map(
      _map_func, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)


def _get_class_representatives_df(embeddings_df: pd.DataFrame,
                                  class_representative_mask: pd.Series,
                                  num_representatives_per_class: int,
                                  rng_key: jax.random.KeyArray) -> pd.DataFrame:
  """Creates a class representatives DataFrame, possibly downsampling at random.

  Args:
    embeddings_df: The embeddings DataFrame.
    class_representative_mask: A boolean mask indicating which embeddings to
      consider for the class representatives.
    num_representatives_per_class: Number of representatives per class to
      select. If 0, an empty DataFrame is returned. If smaller than the number
      of potential representatives indicated by `class_representative_mask`, the
      class representatives are downsampled at random to
      `num_representatives_per_class`.
    rng_key: PRNG key used to perform the random downsampling operation.

  Returns:
    A DataFrame of class representatives.
  """

  num_potential_class_representatives = class_representative_mask.sum()
  if num_representatives_per_class > 0:
    # If needed, downsample to `num_representatives_per_class` at random.
    if num_potential_class_representatives > num_representatives_per_class:
      locations = sorted(
          jax.random.choice(
              rng_key,
              num_potential_class_representatives,
              shape=(num_representatives_per_class,),
              replace=False).tolist())
      # Set all other elements of the mask to False. Since
      # `class_representative_mask` is a boolean series, indexing it with
      # itself returns the True-valued rows. We can then use `locations` to
      # subsample those rows and retrieve the resulting index subset.
      index_subset = class_representative_mask[class_representative_mask].iloc[
          locations].index
      class_representative_mask[~class_representative_mask.index
                                .isin(index_subset)] = False
    return embeddings_df[class_representative_mask]
  else:
    return embeddings_df.iloc[:0, :]


def _get_search_corpus_df(
    embeddings_df: pd.DataFrame, search_corpus_mask: pd.Series,
    class_representatives_df: pd.DataFrame) -> pd.DataFrame:
  """Creates a search corpus DataFrame, excluding any class representative.

  Args:
    embeddings_df: The embeddings DataFrame.
    search_corpus_mask: A boolean mask indicating which embeddings to consider
      for the search corpus.
    class_representatives_df: The class representatives DataFrame.

  Returns:
    A DataFrame of the search corpus.
  """
  search_corpus_df = embeddings_df[search_corpus_mask]
  # Make sure to drop any embeddings present in the class representatives
  # from the search corpus.
  return search_corpus_df.drop(
      class_representatives_df.index,
      # It's possible that the class representatives and the search corpus don't
      # intersect; this is fine.
      errors='ignore')


def _eval_set_generator(
    embeddings_df: pd.DataFrame, eval_set_specification: EvalSetSpecification,
    rng_key: jax.random.KeyArray) -> ClasswiseEvalSetGenerator:
  """Creates a generator for a given eval set.

  Args:
    embeddings_df: A DataFrame containing all evaluation embeddings and their
      relevant metadata.
    eval_set_specification: The specification used to form the eval set.
    rng_key: The PRNG key used to perform random subsampling of the class
      representatives when necessary.

  Yields:
    A (class_name, class_representatives_df, search_corpus_df) tuple.
  """
  global_search_corpus_mask = (
      eval_set_specification.search_corpus_global_mask_fn(embeddings_df))
  global_class_representative_mask = (
      eval_set_specification.class_representative_global_mask_fn(embeddings_df))

  num_representatives_per_class = (
      eval_set_specification.num_representatives_per_class)

  for class_name in eval_set_specification.class_names:
    choice_key, rng_key = jax.random.split(rng_key)

    class_representative_mask = (
        global_class_representative_mask
        & eval_set_specification.class_representative_classwise_mask_fn(
            embeddings_df, class_name))
    search_corpus_mask = (
        global_search_corpus_mask
        & eval_set_specification.search_corpus_classwise_mask_fn(
            embeddings_df, class_name))

    # TODO(vdumoulin): fix the issue upstream to avoid having to skip
    # classes in the first place.
    if class_representative_mask.sum() < num_representatives_per_class:
      logging.warning('Skipping %s as we cannot find enough representatives',
                      class_name)
      continue

    class_representatives_df = _get_class_representatives_df(
        embeddings_df, class_representative_mask, num_representatives_per_class,
        choice_key)

    search_corpus_df = _get_search_corpus_df(embeddings_df, search_corpus_mask,
                                             class_representatives_df)

    # TODO(vdumoulin): fix the issue upstream to avoid having to skip classes
    # in the first place.
    if (search_corpus_df['label'].str.contains(class_name)
        | search_corpus_df['bg_labels'].str.contains(class_name)).sum() == 0:
      logging.warning(
          'Skipping %s as the corpus contains no individual of that class',
          class_name)
      continue

    yield (class_name, class_representatives_df, search_corpus_df)


def _add_dataset_name(features: Dict[str, tf.Tensor],
                      dataset_name: str) -> Dict[str, tf.Tensor]:
  """Adds a 'dataset_name' feature to a features dict.

  Args:
    features: The features dict.
    dataset_name: The 'dataset_name' feature value to add.

  Returns:
    The features dict with the added 'dataset_name' feature.
  """
  features['dataset_name'] = tf.constant(dataset_name)
  if 'bg_labels' not in features:
    features['bg_labels'] = tf.constant('')
  return features


def _create_embeddings_dataframe(
    embedded_datasets: Dict[str, tf.data.Dataset]) -> pd.DataFrame:
  """Builds a dataframe out of all embedded datasets.

  Args:
    embedded_datasets: A mapping from dataset name to embedded dataset.

  Returns:
    The embeddings dataframe.
  """
  # Concatenate all embedded datasets into one embeddings dataset.
  it = iter(embedded_datasets.values())
  embedded_dataset = next(it)
  for dataset in it:
    embedded_dataset = embedded_dataset.concatenate(dataset)

  embeddings_df = pd.DataFrame(embedded_dataset.as_numpy_iterator())

  # Turn 'label', 'bg_labels', 'dataset_name' columns into proper string
  # columns.
  for column_name in ('label', 'bg_labels', 'dataset_name'):
    embeddings_df[column_name] = embeddings_df[column_name].str.decode(
        'utf-8').astype('string')

  return embeddings_df


def prepare_eval_sets(
    config: ConfigDict,
    embedded_datasets: Dict[str, tf.data.Dataset]) -> EvalSetGenerator:
  """Constructs and yields eval sets.

  Args:
    config: The evaluation configuration dict.
    embedded_datasets: A mapping from dataset name to embedded dataset.

  Yields:
    A tuple of (eval_set_name, eval_set_generator). The eval set generator
    itself yields (class_name, class_representatives_df, search_corpus_df)
    tuples. The DataFrame (`*_df`) objects have the following columns:
    - embedding: numpy array of dtype float32.
    - label: space-separated string of foreground labels.
    - bg_labels: space-separated string of background labels.
    - dataset_name: name of the dataset of origin for the embedding.
    - recording_id: integer recording ID.
    - segment_id: integer segment ID within the recording.
  """
  # Add a 'dataset_name' feature to all embedded datasets.
  embedded_datasets = {
      dataset_name: dataset.map(
          functools.partial(_add_dataset_name, dataset_name=dataset_name))
      for dataset_name, dataset in embedded_datasets.items()
  }

  # Build a DataFrame out of all embedded datasets.
  embeddings_df = _create_embeddings_dataframe(embedded_datasets)

  logging.info('Preparing %d unique eval sets.',
               len(config.eval_set_specifications))

  rng_key = jax.random.PRNGKey(config.rng_seed)

  # Yield eval sets one by one.
  for (eval_set_name,
       eval_set_specification) in config.eval_set_specifications.items():
    rng_key, eval_set_key = jax.random.split(rng_key)
    yield eval_set_name, _eval_set_generator(
        embeddings_df=embeddings_df,
        eval_set_specification=eval_set_specification,
        rng_key=eval_set_key)



# TODO(bringingjoy): update return type to a Sequence of
# np.ndarrays when extending create_species_query to support returning multiple
# queries for a given eval species.
def create_averaged_query(
    species_representatives: Sequence[np.ndarray]) -> np.ndarray:
  """Creates a search query from representatives by averaging embeddings.

  Args:
    species_representatives: a collection of vectors representing species
      vocalizations.

  Returns:
    An element-wise average of the vectors to serve as a search query.
  """

  query = np.mean(species_representatives, axis=0)
  return query


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
  """Computes cosine similarity between two vectors and returns the score.

  Args:
    vector_a: an n-dimensional vector of floats.
    vector_b: an n-dimensional vector of floats.

  Returns:
    The cosine similarity score between two vectors A and B, where increasing
    score corresponds to vector similarity.

  Note:
    Definition: A dot B / ||A|| * ||B||. Scores
    close to -1 means A and B are 'opposite' vectors,
    close to 0 means A and B are 'orthogonal' vectors, and
    close to 1 means to A and B are very similar vectors.
  """

  dot_prod = np.dot(vector_a, vector_b)
  norm_prod = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

  return dot_prod / norm_prod
