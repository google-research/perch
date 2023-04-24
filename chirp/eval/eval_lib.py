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

"""Utility functions for evaluation."""

import dataclasses
import functools
import os
from typing import (Callable, Generator, Mapping, Sequence, TypeVar)

from absl import logging
from chirp.data import utils as data_utils
from chirp.models import metrics
from chirp.taxonomy import namespace_db
import jax
import ml_collections
import numpy as np
import pandas as pd
import tensorflow as tf

_EMBEDDING_KEY = 'embedding'
_LABEL_KEY = 'label'
_BACKGROUND_KEY = 'bg_labels'
ConfigDict = ml_collections.ConfigDict

MaskFunction = Callable[[pd.DataFrame], pd.Series]
ClasswiseMaskFunction = Callable[[pd.DataFrame, str], pd.Series]
ClasswiseEvalSetGenerator = Generator[
    tuple[str, pd.DataFrame, pd.DataFrame], None, None
]
EvalSetGenerator = Generator[tuple[str, ClasswiseEvalSetGenerator], None, None]
EvalModelCallable = Callable[[np.ndarray], np.ndarray]

_T = TypeVar('_T', bound='EvalSetSpecification')


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
      uniformly at random to the correct size. If -1, all representatives are
      used.
  """

  class_names: Sequence[str]
  search_corpus_global_mask_fn: MaskFunction
  search_corpus_classwise_mask_fn: ClasswiseMaskFunction
  class_representative_global_mask_fn: MaskFunction
  class_representative_classwise_mask_fn: ClasswiseMaskFunction
  num_representatives_per_class: int

  @classmethod
  def v1_specification(
      cls: type[_T],
      location: str,
      corpus_type: str,
      num_representatives_per_class: int,
  ) -> _T:
    """Instantiates an eval protocol v1 EvalSetSpecification.

    Args:
      location: Geographical location in {'ssw', 'colombia', 'hawaii'}.
      corpus_type: Corpus type in {'xc_fg', 'xc_bg', 'birdclef'}.
      num_representatives_per_class: Number of class representatives to sample.
        If -1, all representatives are used.

    Returns:
      The EvalSetSpecification.
    """
    downstream_class_names = (
        namespace_db.load_db().class_lists['downstream_species'].classes
    )
    # "At-risk" species are excluded from downstream data due to conservation
    # status.
    class_names = {
        'ssw': (
            namespace_db.load_db()
            .class_lists['artificially_rare_species']
            .classes
        ),
        'colombia': [
            c
            for c in namespace_db.load_db()
            .class_lists['birdclef2019_colombia']
            .classes
            if c in downstream_class_names
        ],
        'hawaii': [
            c
            for c in namespace_db.load_db().class_lists['hawaii'].classes
            if c in downstream_class_names
        ],
    }[location]

    # The name of the dataset to draw embeddings from to form the corpus.
    corpus_dataset_name = (
        f'birdclef_{location}' if corpus_type == 'birdclef' else 'xc_downstream'
    )
    has_corpus_dataset_name = (
        lambda df: df['dataset_name'] == corpus_dataset_name
    )
    # Only include embeddings in the searchcorpus which have foreground
    # ('label') and/or background labels ('bg_labels') for some class in
    # `class_names`, which are encoded as space-separated species IDs/codes.
    has_some_fg_annotation = (
        # `'|'.join(class_names)` is a regex which matches *any* class in
        # `class_names`.
        lambda df: df['label'].str.contains('|'.join(class_names))
    )
    has_some_bg_annotation = lambda df: df['bg_labels'].str.contains(
        '|'.join(class_names)
    )
    has_some_annotation = lambda df: has_some_fg_annotation(
        df
    ) | has_some_bg_annotation(df)

    class_representative_dataset_name = {
        'ssw': 'xc_artificially_rare',
        'colombia': 'xc_downstream',
        'hawaii': 'xc_downstream',
    }[location]

    return cls(
        class_names=class_names,
        search_corpus_global_mask_fn=(
            lambda df: has_corpus_dataset_name(df) & has_some_annotation(df)
        ),
        # Ensure that target species' background vocalizations are not present
        # in the 'xc_fg' corpus and vice versa.
        search_corpus_classwise_mask_fn={
            'xc_fg': lambda df, n: ~df['bg_labels'].str.contains(n),
            'xc_bg': lambda df, n: ~df['label'].str.contains(n),
            'birdclef': lambda df, _: df['label'].map(lambda s: True),
        }[corpus_type],
        # Class representatives are drawn only from foreground-vocalizing
        # species present in Xeno-Canto.
        class_representative_global_mask_fn=(
            lambda df: df['dataset_name'] == class_representative_dataset_name
        ),
        class_representative_classwise_mask_fn=(
            lambda df, class_name: df['label'].str.contains(class_name)
        ),
        num_representatives_per_class=num_representatives_per_class,
    )

  @classmethod
  def v2_specification(
      cls: type[_T],
      location: str,
      corpus_type: str,
      num_representatives_per_class: int,
  ) -> _T:
    """Instantiates an eval protocol v2 EvalSetSpecification.

    Args:
      location: Geographical location in {'ssw', 'colombia', 'hawaii'}.
      corpus_type: Corpus type in {'xc_fg', 'xc_bg', 'birdclef'}.
      num_representatives_per_class: Number of class representatives to sample.
        If -1, all representatives are used.

    Returns:
      The EvalSetSpecification.
    """
    downstream_class_names = (
        namespace_db.load_db().class_lists['downstream_species'].classes
    )
    # "At-risk" species are excluded from downstream data due to conservation
    # status.
    class_names = {
        'ssw': (
            namespace_db.load_db()
            .class_lists['artificially_rare_species']
            .classes
        ),
        'colombia': [
            c
            for c in namespace_db.load_db()
            .class_lists['birdclef2019_colombia']
            .classes
            if c in downstream_class_names
        ],
        'hawaii': [
            c
            for c in namespace_db.load_db().class_lists['hawaii'].classes
            if c in downstream_class_names
        ],
    }[location]

    # The name of the dataset to draw embeddings from to form the corpus.
    corpus_dataset_name = (
        f'birdclef_{location}' if corpus_type == 'birdclef' else 'xc_downstream'
    )
    has_corpus_dataset_name = (
        lambda df: df['dataset_name'] == corpus_dataset_name
    )
    # Only include embeddings in the searchcorpus which have foreground
    # ('label') and/or background labels ('bg_labels') for some class in
    # `class_names`, which are encoded as space-separated species IDs/codes.
    has_some_fg_annotation = (
        # `'|'.join(class_names)` is a regex which matches *any* class in
        # `class_names`.
        lambda df: df['label'].str.contains('|'.join(class_names))
    )
    has_some_bg_annotation = lambda df: df['bg_labels'].str.contains(
        '|'.join(class_names)
    )
    has_some_annotation = lambda df: has_some_fg_annotation(
        df
    ) | has_some_bg_annotation(df)

    class_representative_dataset_name = {
        'ssw': 'xc_artificially_rare_class_reps',
        'colombia': 'xc_downstream_class_reps',
        'hawaii': 'xc_downstream_class_reps',
    }[location]

    return cls(
        class_names=class_names,
        search_corpus_global_mask_fn=(
            lambda df: has_corpus_dataset_name(df) & has_some_annotation(df)
        ),
        # Ensure that target species' background vocalizations are not present
        # in the 'xc_fg' corpus and vice versa.
        search_corpus_classwise_mask_fn={
            'xc_fg': lambda df, n: ~df['bg_labels'].str.contains(n),
            'xc_bg': lambda df, n: ~df['label'].str.contains(n),
            'birdclef': lambda df, _: df['label'].map(lambda s: True),
        }[corpus_type],
        # Class representatives are drawn from foreground-vocalizing species
        # present in Xeno-Canto after applying peak-finding.
        class_representative_global_mask_fn=(
            lambda df: df['dataset_name'] == class_representative_dataset_name
        ),
        class_representative_classwise_mask_fn=(
            lambda df, class_name: df['label'].str.contains(class_name)
        ),
        num_representatives_per_class=num_representatives_per_class,
    )


def _load_eval_dataset(dataset_config: ConfigDict) -> tf.data.Dataset:
  """Loads an evaluation dataset from its corresponding configuration dict."""
  return data_utils.get_dataset(
      split=dataset_config.split,
      is_train=False,
      dataset_directory=dataset_config.tfds_name,
      tfds_data_dir=dataset_config.tfds_data_dir,
      tf_data_service_address=None,
      pipeline=dataset_config.pipeline,
  )[0]


def load_eval_datasets(config: ConfigDict) -> dict[str, tf.data.Dataset]:
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
    model_callback: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
) -> tf.data.Dataset:
  """Embeds the audio slice in each tf.Example across the input dataset.

  Args:
    dataset: A TF Dataset composed of tf.Examples.
    model_callback: A Callable that takes a batched NumPy array and produces a
      batched embedded NumPy array.
    batch_size: The number of examples to embed in each batch.

  Returns:
    An updated TF Dataset with a new 'embedding' feature and deleted 'audio'
    feature.
  """

  def _map_func(example):
    example['embedding'] = tf.numpy_function(
        func=model_callback, inp=[example['audio']], Tout=tf.float32
    )
    del example['audio']
    return example

  # Use the 'audio' feature to produce a model embedding; delete the old 'audio'
  # feature.
  embedded_dataset = (
      dataset.batch(batch_size, drop_remainder=False).prefetch(1).map(_map_func)
  )

  return embedded_dataset.unbatch()


def _get_class_representatives_df(
    embeddings_df: pd.DataFrame,
    class_representative_mask: pd.Series,
    num_representatives_per_class: int,
    rng_key: jax.random.KeyArray,
) -> pd.DataFrame:
  """Creates a class representatives DataFrame, possibly downsampling at random.

  Args:
    embeddings_df: The embeddings DataFrame.
    class_representative_mask: A boolean mask indicating which embeddings to
      consider for the class representatives.
    num_representatives_per_class: Number of representatives per class to
      select. If -1, all representatives are returned. When the number of
      representatives indicated by `class_representative_mask` is greater than
      `num_representatives_per_class`, they are downsampled at random to that
      threshold.
    rng_key: PRNG key used to perform the random downsampling operation.

  Returns:
    A DataFrame of class representatives.
  """

  num_potential_class_representatives = class_representative_mask.sum()
  if num_representatives_per_class >= 0:
    # If needed, downsample to `num_representatives_per_class` at random.
    if num_potential_class_representatives > num_representatives_per_class:
      locations = sorted(
          jax.random.choice(
              rng_key,
              num_potential_class_representatives,
              shape=(num_representatives_per_class,),
              replace=False,
          ).tolist()
      )
      # Set all other elements of the mask to False. Since
      # `class_representative_mask` is a boolean series, indexing it with
      # itself returns the True-valued rows. We can then use `locations` to
      # subsample those rows and retrieve the resulting index subset.
      index_subset = (
          class_representative_mask[class_representative_mask]
          .iloc[locations]
          .index
      )
      class_representative_mask = class_representative_mask.copy()
      class_representative_mask[
          ~class_representative_mask.index.isin(index_subset)
      ] = False
  return embeddings_df[class_representative_mask]


def _get_search_corpus_df(
    embeddings_df: pd.DataFrame,
    search_corpus_mask: pd.Series,
    class_representatives_df: pd.DataFrame,
) -> pd.DataFrame:
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
  #
  # Note that for eval v2, indices will not be dropped because class reps &
  # search corpus examples are drawn from different datasets. For XC, there will
  # be one match between the class rep & search corpus example for each species.
  return search_corpus_df.drop(
      class_representatives_df.index,
      # It's possible that the class representatives and the search corpus don't
      # intersect; this is fine.
      errors='ignore',
  )


def _eval_set_generator(
    embeddings_df: pd.DataFrame,
    eval_set_specification: EvalSetSpecification,
    rng_key: jax.random.KeyArray,
) -> ClasswiseEvalSetGenerator:
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
      eval_set_specification.search_corpus_global_mask_fn(embeddings_df)
  )
  global_class_representative_mask = (
      eval_set_specification.class_representative_global_mask_fn(embeddings_df)
  )

  num_representatives_per_class = (
      eval_set_specification.num_representatives_per_class
  )

  for class_name in eval_set_specification.class_names:
    choice_key, rng_key = jax.random.split(rng_key)

    class_representative_mask = (
        global_class_representative_mask
        & eval_set_specification.class_representative_classwise_mask_fn(
            embeddings_df, class_name
        )
    )
    search_corpus_mask = (
        global_search_corpus_mask
        & eval_set_specification.search_corpus_classwise_mask_fn(
            embeddings_df, class_name
        )
    )

    # TODO(vdumoulin): fix the issue upstream to avoid having to skip
    # classes in the first place.
    if (
        num_representatives_per_class >= 0
        and class_representative_mask.sum() < num_representatives_per_class
    ):
      logging.warning(
          'Skipping %s as we cannot find enough representatives', class_name
      )
      continue

    class_representatives_df = _get_class_representatives_df(
        embeddings_df,
        class_representative_mask,
        num_representatives_per_class,
        choice_key,
    )

    search_corpus_df = _get_search_corpus_df(
        embeddings_df, search_corpus_mask, class_representatives_df
    )

    # TODO(vdumoulin): fix the issue upstream to avoid having to skip classes
    # in the first place.
    if (
        search_corpus_df['label'].str.contains(class_name)
        | search_corpus_df['bg_labels'].str.contains(class_name)
    ).sum() == 0:
      logging.warning(
          'Skipping %s as the corpus contains no individual of that class',
          class_name,
      )
      continue

    yield (class_name, class_representatives_df, search_corpus_df)


def _add_dataset_name(
    features: dict[str, tf.Tensor], dataset_name: str
) -> dict[str, tf.Tensor]:
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


def _numpy_iterator_with_progress_logging(embedded_dataset):

  for i, example in enumerate(embedded_dataset.as_numpy_iterator()):
    yield example
    logging.log_every_n(
        logging.INFO,
        'Converting concatenated embedded dataset to dataframe (%d done)...',
        1000,  # n=1000
        i,
    )


def _create_embeddings_dataframe(
    embedded_datasets: dict[str, tf.data.Dataset], config: ConfigDict
) -> pd.DataFrame:
  """Builds a dataframe out of all embedded datasets.

  The dataframe also contains upstream class representations (rows with the
  'learned_representations' value for their 'dataset_name' column).

  Args:
    embedded_datasets: A mapping from dataset name to embedded dataset.
    config: The evaluation configuration dict.

  Returns:
    The embeddings dataframe, with additional rows for upstream class
    representations (accessible through `embeddings_df[
    embeddings_df['dataset_name'] == 'learned_representations']`).
  """
  # Concatenate all embedded datasets into one embeddings dataset.
  it = iter(embedded_datasets.values())
  embedded_dataset = next(it)
  for dataset in it:
    embedded_dataset = embedded_dataset.concatenate(dataset)

  if config.debug.embedded_dataset_cache_path:
    embedded_dataset = embedded_dataset.cache(
        config.debug.embedded_dataset_cache_path
    )

  embeddings_df = pd.DataFrame(
      _numpy_iterator_with_progress_logging(embedded_dataset)
  )

  # Encode 'label', 'bg_labels', 'dataset_name' column data as strings.
  for column_name in ('label', 'bg_labels', 'dataset_name'):
    embeddings_df[column_name] = (
        embeddings_df[column_name].str.decode('utf-8').astype('string')
    )

  return embeddings_df


def prepare_eval_sets(
    config: ConfigDict, embedded_datasets: dict[str, tf.data.Dataset]
) -> EvalSetGenerator:
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
          functools.partial(_add_dataset_name, dataset_name=dataset_name)
      )
      for dataset_name, dataset in embedded_datasets.items()
  }

  # Build a DataFrame out of all embedded datasets.
  embeddings_df = _create_embeddings_dataframe(embedded_datasets, config)

  logging.info(
      'Preparing %d unique eval sets.', len(config.eval_set_specifications)
  )

  rng_key = jax.random.PRNGKey(config.rng_seed)

  # Yield eval sets one by one.
  for (
      eval_set_name,
      eval_set_specification,
  ) in config.eval_set_specifications.items():
    rng_key, eval_set_key = jax.random.split(rng_key)
    yield eval_set_name, _eval_set_generator(
        embeddings_df=embeddings_df,
        eval_set_specification=eval_set_specification,
        rng_key=eval_set_key,
    )


def search(
    eval_and_search_corpus: ClasswiseEvalSetGenerator,
    learned_representations: Mapping[str, np.ndarray],
    create_species_query: Callable[[Sequence[np.ndarray]], np.ndarray],
    search_score: Callable[[np.ndarray, np.ndarray], float],
) -> Mapping[str, pd.DataFrame]:
  """Performs search over evaluation set examples and search corpus pairs.

  Args:
    eval_and_search_corpus: A ClasswiseEvalSetGenerator, an alias for a
      Generator of Tuples, where each contains an eval set species name
      (class_name), a DataFrame containing a collection of representatives of
      the eval set species, and a DataFrame containing a collection of search
      corpus species examples to perform search over.
    learned_representations: Mapping from class name to its learned
      representation. If a key exists in the mapping, the corresponding
      representation is used instead of calling `create_species_query` on the
      class representatives.
    create_species_query: A function callback provided by the user to construct
      a search query from a collection of species vectors. Choice of methodology
      left up to the user.
    search_score: A function callback provided by the user to produce a score by
      comparing two vectors (e.g. query and species representative/embedding).

  Returns:
    A mapping of query-species ID to a DataFrame of search results. The results
    DataFrame is structured as follows, with num(search_corpus) rows and two
    columns:
    - each row corresponds to the results for a single search corpus example,
    - column 1 contains a search score (float)
    - column 2 contains an indicator of whether the eval and search species are
    the same (bool).
  """

  # A mapping from eval species class to a DataFrame of search results.
  eval_search_results = dict()

  for species_id, eval_reps, search_corpus in eval_and_search_corpus:
    if species_id in learned_representations:
      query = learned_representations[species_id]
    else:
      query = create_species_query(eval_reps['embedding'])

    species_scores = query_search(
        query=query,
        species_id=species_id,
        search_corpus=search_corpus,
        search_score=search_score,
    )

    eval_search_results[species_id] = species_scores

  return eval_search_results


def query_search(
    query: np.ndarray,
    species_id: str,
    search_corpus: pd.DataFrame,
    search_score: Callable[[np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
  """Performs vector-based comparison between the query and search corpus.

  Args:
    query: A vector representation of an evaluation species.
    species_id: The species ID of the query.
    search_corpus: A DataFrame containing rows of search examples.
    search_score: A Callable that operates over two vectors and returns a float.

  Returns:
    A DataFrame where each row corresponds to the results on a single search
    examplar, with columns for a numeric score and species match (bool) checked
    between the query species ID and the search corpus examplar's foreground and
    background species labels.
  """

  search_species_scores = pd.DataFrame()
  search_species_scores['score'] = search_corpus[_EMBEDDING_KEY].apply(
      lambda x: search_score(query, x)
  )
  fg_species_match = (
      search_corpus[_LABEL_KEY]
      .apply(lambda x: species_id in x.split(' '))
      .astype(np.int16)
  )
  bg_species_match = (
      search_corpus[_BACKGROUND_KEY]
      .apply(lambda x: species_id in x.split(' '))
      .astype(np.int16)
  )
  search_species_scores['species_match'] = fg_species_match | bg_species_match

  return search_species_scores


def compute_metrics(
    eval_set_name: str,
    eval_set_results: Mapping[str, pd.DataFrame],
    sort_descending: bool = True,
):
  """Computes roc-auc & average precision on provided eval results DataFrame.

  Args:
    eval_set_name: The name of the evaluation set.
    eval_set_results: A mapping from species ID to a DataFrame of the search
      results for that species (with columns 'score' and 'species_match').
    sort_descending: An indicator if the search result ordering is in descending
      order to be used post-search by average-precision based metrics. Sorts in
      descending order by default.

  Returns:
    Produces metrics (roc-auc & average precision) computed for each species in
    the given eval set and writes these to a csv for each eval set.
  """

  species_metric_eval_set = list()
  for eval_species, eval_results in eval_set_results.items():
    eval_scores = eval_results['score'].values
    species_label_match = eval_results['species_match'].values

    roc_auc = metrics.roc_auc(
        logits=eval_scores.reshape(-1, 1),
        labels=species_label_match.reshape(-1, 1),
        sort_descending=sort_descending,
    )[
        'macro'
    ]  # Dictionary of macro, geometric, individual & individual_var.

    average_precision = metrics.average_precision(
        scores=eval_scores,
        labels=species_label_match,
        sort_descending=sort_descending,
    )
    species_metric_eval_set.append(
        (eval_species, average_precision, roc_auc, eval_set_name)
    )

  return species_metric_eval_set


def write_results_to_csv(
    metric_results: Sequence[tuple[str, float, float, str]],
    write_results_dir: str,
):
  """Write evaluation metric results to csv.

    Writes a csv file where each row corresponds to a particular evaluation
    example's search task performance. If the provided write_results_dir doesn't
    exist, it is created. If an evaluation results file already exists, it is
    overwritten.

  Args:
    metric_results: A sequence of tuples of (eval species name,
      average_precision, roc_auc [arithmetic mean], evaluation set name) to
      write to csv. The first row encodes the column header or column names.
    write_results_dir: The path to write the computed metrics to file.
  """

  write_results_path = os.path.join(write_results_dir, 'evaluation_results.csv')
  results_df = pd.DataFrame(metric_results[1:], columns=metric_results[0])

  # Check if the specified directory exists; if not, create & write to csv.
  if write_results_dir.find('cns') == 0:
    if not os.path.exists(write_results_dir):
      os.makedirs(write_results_dir)
    results_df.to_csv(write_results_path, index=False)



# TODO(bringingjoy): update return type to a Sequence of
# np.ndarrays when extending create_species_query to support returning multiple
# queries for a given eval species.
def create_averaged_query(
    species_representatives: Sequence[np.ndarray],
) -> np.ndarray:
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
