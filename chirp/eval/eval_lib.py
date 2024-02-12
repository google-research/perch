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

"""Utility functions for evaluation."""

import dataclasses
import functools
import os
from typing import Callable, Iterator, Mapping, Sequence, TypeVar

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

EvalModelCallable = Callable[[np.ndarray], np.ndarray]
_T = TypeVar('_T', bound='EvalSetSpecification')
_EVAL_REGIONS = (
    'ssw',
    'coffee_farms',
    'hawaii',
    'high_sierras',
    'sierras_kahl',  # Sierra Nevada
    'peru',
)


# TODO(bringingjoy): Update once mismatched species codes are resolved in our
# class lists.
_MISMATCHED_SPECIES_CODES = [
    'reevir1',
    'gnwtea',
    'grnjay',
    'butwoo1',
    'unknown',
]


@dataclasses.dataclass
class EvalSetSpecification:
  """A specification for an eval set.

  Attributes:
    class_names: Class names over which to perform the evaluation.
    search_corpus_global_mask_expr: String expression passed to the embeddings
      dataframe's `eval` method to obtain a boolean mask over its rows. Used to
      represent global properties like `df['dataset_name'] == 'coffee_farms'`.
      Computed once and combined with `search_corpus_classwise_mask_fn` for
      every class in `class_names` to perform boolean indexing on the embeddings
      dataframe and form the search corpus.
    search_corpus_classwise_mask_fn: Function mapping a class name to a string
      expression passed to the embeddings dataframe's `eval` method to obtain a
      boolean mask over its rows. Used to represent classwise properties like
      `~df['bg_labels'].str.contains(class_name)`. Combined with
      `search_corpus_global_mask_expr` for every class in `class_names` to
      perform boolean indexing on the embeddings dataframe and form the search
      corpus.
    class_representative_global_mask_expr: String expression passed to the
      embeddings dataframe's `eval` method to obtain a boolean mask over its
      rows. Used to represent global properties like `df['dataset_name'] ==
      'xc_downstream'`. Computed once and combined with
      `class_representative_corpus_classwise_mask_fn` for every class in
      `class_names` to perform boolean indexing on the embeddings dataframe and
      form the collection of class representatives.
    class_representative_classwise_mask_fn: Function mapping a class name to a
      string expression passed to the embeddings dataframe's `eval` method to
      obtain a boolean mask over its rows. Used to represent classwise
      properties like `df['label'].str.contains(class_name)`. Combined with
      `class_representative_corpus_global_mask_expr` for every class in
      `class_names` to perform boolean indexing on the embeddings dataframe and
      form the collection of class representatives.
    num_representatives_per_class: Number of class representatives to sample. If
      the pool of potential representatives is larger, it's downsampled
      uniformly at random to the correct size. If -1, all representatives are
      used.
  """

  class_names: Sequence[str]
  search_corpus_global_mask_expr: str
  search_corpus_classwise_mask_fn: Callable[[str], str]
  class_representative_global_mask_expr: str
  class_representative_classwise_mask_fn: Callable[[str], str]
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
      location: Geographical location in {'ssw', 'coffee_farms', 'hawaii'}.
      corpus_type: Corpus type in {'xc_fg', 'xc_bg', 'birdclef'}.
      num_representatives_per_class: Number of class representatives to sample.
        If -1, all representatives are used.

    Returns:
      The EvalSetSpecification.
    """
    downstream_class_names = (
        namespace_db.load_db().class_lists['downstream_species_v2'].classes
    )
    # "At-risk" species are excluded from downstream data due to conservation
    # status.
    class_names = {
        'ssw': (
            namespace_db.load_db()
            .class_lists['artificially_rare_species_v2']
            .classes
        ),
        'coffee_farms': [
            c
            for c in namespace_db.load_db().class_lists['coffee_farms'].classes
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

    class_representative_dataset_name = {
        'ssw': 'xc_artificially_rare_v2',
        'coffee_farms': 'xc_downstream',
        'hawaii': 'xc_downstream',
    }[location]

    # `'|'.join(class_names)` is a regex which matches *any* class in
    # `class_names`.
    class_name_regexp = '|'.join(class_names)

    return cls(
        class_names=class_names,
        # Only include embeddings in the search corpus which have foreground
        # ('label') and/or background labels ('bg_labels') for some class in
        # `class_names`, which are encoded as space-separated species IDs/codes.
        search_corpus_global_mask_expr=(
            f'dataset_name == "{corpus_dataset_name}" and '
            f'(label.str.contains("{class_name_regexp}") or '
            f'bg_labels.str.contains("{class_name_regexp}"))'
        ),
        # Ensure that target species' background vocalizations are not present
        # in the 'xc_fg' corpus and vice versa.
        search_corpus_classwise_mask_fn={
            'xc_fg': lambda name: f'not bg_labels.str.contains("{name}")',
            'xc_bg': lambda name: f'not label.str.contains("{name}")',
            'birdclef': lambda _: 'label.str.contains("")',
        }[corpus_type],
        # Class representatives are drawn only from foreground-vocalizing
        # species present in Xeno-Canto.
        class_representative_global_mask_expr=(
            f'dataset_name == "{class_representative_dataset_name}"'
        ),
        class_representative_classwise_mask_fn=(
            lambda name: f'label.str.contains("{name}")'
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
      location: Geographical location in {'ssw', 'coffee_farms', 'hawaii'}.
      corpus_type: Corpus type in {'xc_fg', 'xc_bg', 'soundscapes'}.
      num_representatives_per_class: Number of class representatives to sample.
        If -1, all representatives are used.

    Returns:
      The EvalSetSpecification.
    """
    downstream_class_names = (
        namespace_db.load_db().class_lists['downstream_species_v2'].classes
    )
    class_names = {}
    for region in _EVAL_REGIONS:
      if region == 'ssw':
        # Filter recordings with 'unknown' species label.
        ssw = 'artificially_rare_species_v2'
        species = [
            c
            for c in namespace_db.load_db().class_lists[ssw].classes
            if c not in _MISMATCHED_SPECIES_CODES
        ]
      elif region in ('peru', 'high_sierras', 'sierras_kahl'):
        species = [
            c
            for c in namespace_db.load_db().class_lists[region].classes
            if c not in _MISMATCHED_SPECIES_CODES
        ]
      else:
        # Keep recordings which map to downstream class species.
        species = [
            c
            for c in namespace_db.load_db().class_lists[region].classes
            if c in downstream_class_names
            and c not in _MISMATCHED_SPECIES_CODES
        ]
      class_names[region] = species
    class_names = class_names[location]

    # The name of the dataset to draw embeddings from to form the corpus.
    corpus_dataset_name = (
        f'soundscapes_{location}'
        if corpus_type == 'soundscapes'
        else 'xc_downstream'
    )

    class_representative_dataset_name = 'xc_class_reps'
    class_name_regexp = '|'.join(class_names)

    return cls(
        class_names=class_names,
        # Only include embeddings in the search corpus which have foreground
        # ('label') and/or background labels ('bg_labels') for some class in
        # `class_names`, which are encoded as space-separated species IDs/codes.
        search_corpus_global_mask_expr=(
            f'dataset_name == "{corpus_dataset_name}" and '
            f'(label.str.contains("{class_name_regexp}") or '
            f'bg_labels.str.contains("{class_name_regexp}"))'
        ),
        # Ensure that target species' background vocalizations are not present
        # in the 'xc_fg' corpus and vice versa.
        search_corpus_classwise_mask_fn={
            'xc_fg': lambda name: f'not bg_labels.str.contains("{name}")',
            'xc_bg': lambda name: f'not label.str.contains("{name}")',
            'soundscapes': lambda _: 'label.str.contains("")',
        }[corpus_type],
        # Class representatives are drawn from foreground-vocalizing species
        # present in Xeno-Canto after applying peak-finding.
        class_representative_global_mask_expr=(
            f'dataset_name == "{class_representative_dataset_name}"'
        ),
        class_representative_classwise_mask_fn=(
            lambda name: f'label.str.contains("{name}")'
        ),
        num_representatives_per_class=num_representatives_per_class,
    )


@dataclasses.dataclass
class ClasswiseEvalSet:
  class_name: str
  class_representatives_df: pd.DataFrame
  search_corpus_mask: pd.Series


@dataclasses.dataclass
class EvalSet:
  name: str
  search_corpus_df: pd.DataFrame
  classwise_eval_sets: tuple[ClasswiseEvalSet, ...]


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
    rng_key: jax.Array,
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


@dataclasses.dataclass
class _HashedEmbeddingsDataFrame:
  """A hashable dataclass to encapsulate an embeddings DataFrame.

  NOTE: The hash implementation relies on a unique object ID for the DataFrame,
  which is determined at creation time. This is fast, but brittle. The
  embeddings DataFrame should *never* be modified in-place; doing so would
  result in a different DataFrame with the same hash.
  """

  df: pd.DataFrame

  def __hash__(self):
    return id(self.df)


@functools.cache
def _df_eval(hashable_df: _HashedEmbeddingsDataFrame, expr: str) -> pd.Series:
  return hashable_df.df.eval(expr, engine='python')


def _prepare_eval_set(
    embeddings_df: _HashedEmbeddingsDataFrame,
    eval_set_specification: EvalSetSpecification,
    rng_key: jax.Array,
) -> tuple[pd.DataFrame, tuple[ClasswiseEvalSet, ...]]:
  """Prepares a single eval set.

  This entails creating and returning a search corpus DataFrame and a classwise
  eval set generator. The latter iterates over classes in the eval set
  specification and yields (class_name, class_representatives_df,
  search_corpus_mask) tuples. Each search corpus mask indicates which part of
  the search corpus should be ignored in the context of its corresponding class
  (e.g., because it overlaps with the chosen class representatives).

  Args:
    embeddings_df: A DataFrame containing all evaluation embeddings and their
      relevant metadata.
    eval_set_specification: The specification used to form the eval set.
    rng_key: The PRNG key used to perform random subsampling of the class
      representatives when necessary.

  Returns:
    A (search_corpus_df, classwise_eval_set_generator) tuple.
  """
  global_search_corpus_mask = _df_eval(
      embeddings_df, eval_set_specification.search_corpus_global_mask_expr
  )
  global_class_representative_mask = _df_eval(
      embeddings_df,
      eval_set_specification.class_representative_global_mask_expr,
  )

  num_representatives_per_class = (
      eval_set_specification.num_representatives_per_class
  )

  search_corpus_df = embeddings_df.df[global_search_corpus_mask]
  classwise_eval_sets = []

  for class_name in eval_set_specification.class_names:
    choice_key, rng_key = jax.random.split(rng_key)

    class_representative_mask = global_class_representative_mask & _df_eval(
        embeddings_df,
        eval_set_specification.class_representative_classwise_mask_fn(
            class_name
        ),
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
        embeddings_df.df,
        class_representative_mask,
        num_representatives_per_class,
        choice_key,
    )

    search_corpus_mask = (
        global_search_corpus_mask
        & _df_eval(
            embeddings_df,
            eval_set_specification.search_corpus_classwise_mask_fn(class_name),
        )
        # Exclude rows selected as class representatives.
        & ~embeddings_df.df.index.isin(class_representatives_df.index)
    )
    search_corpus_mask = search_corpus_mask.loc[search_corpus_df.index]

    # TODO(vdumoulin): fix the issue upstream to avoid having to skip classes
    # in the first place.
    if (
        search_corpus_df['label'][search_corpus_mask].str.contains(class_name)
        | search_corpus_df['bg_labels'][search_corpus_mask].str.contains(
            class_name
        )
    ).sum() == 0:
      logging.warning(
          'Skipping %s as the corpus contains no individual of that class',
          class_name,
      )
      continue

    classwise_eval_sets.append(
        ClasswiseEvalSet(
            class_name=class_name,
            class_representatives_df=class_representatives_df,
            search_corpus_mask=search_corpus_mask,
        )
    )

  return search_corpus_df, tuple(classwise_eval_sets)


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
) -> Iterator[EvalSet]:
  """Constructs and yields eval sets.

  Args:
    config: The evaluation configuration dict.
    embedded_datasets: A mapping from dataset name to embedded dataset.

  Yields:
    A tuple of (eval_set_name, search_corpus_df, classwise_eval_set_generator).
    The classwise eval set generator itself yields (class_name,
    class_representatives_df, search_corpus_mask) tuples. Each search corpus
    mask indicates which part of the search corpus should be ignored in the
    context of its corresponding class (e.g., because it overlaps with the
    chosen class representatives). The DataFrame (`*_df`) objects have the
    following columns:
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
    search_corpus_df, classwise_eval_sets = _prepare_eval_set(
        embeddings_df=_HashedEmbeddingsDataFrame(embeddings_df),
        eval_set_specification=eval_set_specification,
        rng_key=eval_set_key,
    )
    yield EvalSet(
        name=eval_set_name,
        search_corpus_df=search_corpus_df,
        classwise_eval_sets=classwise_eval_sets,
    )


def search(
    eval_set: EvalSet,
    learned_representations: Mapping[str, np.ndarray],
    create_species_query: Callable[[Sequence[np.ndarray]], np.ndarray],
    search_score: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Mapping[str, pd.DataFrame]:
  """Performs search over evaluation set examples and search corpus pairs.

  Args:
    eval_set: The evaluation set over which to perform search.
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
    - column 3 contains an indicator of whether to exclude the row for the
    search corpus (bool).
  """

  # A mapping from eval species class to a DataFrame of search results.
  eval_search_results = dict()

  # Gather all query vectors.
  queries = np.stack(
      [
          learned_representations[ces.class_name]
          if ces.class_name in learned_representations
          else create_species_query(ces.class_representatives_df['embedding'])
          for ces in eval_set.classwise_eval_sets
      ]
  )

  # Perform a matrix-matrix scoring using stacked queries and search corpus
  # embeddings.
  scores = search_score(
      queries, np.stack(eval_set.search_corpus_df[_EMBEDDING_KEY].tolist())
  )

  for score, ces in zip(scores, eval_set.classwise_eval_sets):
    species_scores = _make_species_scores_df(
        score=pd.Series(score.tolist(), index=eval_set.search_corpus_df.index),
        species_id=ces.class_name,
        search_corpus=eval_set.search_corpus_df,
        search_corpus_mask=ces.search_corpus_mask,
    )

    eval_search_results[ces.class_name] = species_scores

  return eval_search_results


def _make_species_scores_df(
    score: pd.Series,
    species_id: str,
    search_corpus: pd.DataFrame,
    search_corpus_mask: pd.Series,
) -> pd.DataFrame:
  """Creates a DataFrame of scores and other metric-relevant information.

  Args:
    score: A Series of scores (with respect to a query for species `species_id`)
      for each embedding in the search corpus.
    species_id: The species ID of the query.
    search_corpus: A DataFrame containing rows of search examples.
    search_corpus_mask: A boolean Series indicating which part of the search
      corpus should be ignored in the context of its corresponding class (e.g.,
      because it overlaps with the chosen class representatives).

  Returns:
    A DataFrame where each row corresponds to the results on a single search
    examplar, with columns for i) a numeric score, ii) a species match (bool)
    checked between the query species ID and the search corpus examplar's
    foreground and background species labels, and iii) a label mask (bool)
    indicating whether the row should be ignored when computing metrics.
  """

  search_species_scores = pd.DataFrame()
  search_species_scores['score'] = score
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
  search_species_scores['label_mask'] = search_corpus_mask

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
      results for that species (with columns 'score', 'species_match', and
      'label_mask').
    sort_descending: An indicator if the search result ordering is in descending
      order to be used post-search by average-precision based metrics. Sorts in
      descending order by default.

  Returns:
    Produces metrics (average_precision, roc_auc, num_pos_match, num_neg_match)
    computed for each species in the given eval set and writes these to a csv
    for each eval set.
  """
  # TODO(hamer): consider moving eval_set_name metadata (i.e. # exemplars, seed)
  # to separate columns in the metric results.
  species_metric_eval_set = list()
  for eval_species, eval_results in eval_set_results.items():
    eval_scores = eval_results['score'].to_numpy()
    species_label_match = eval_results['species_match'].to_numpy()
    label_mask = eval_results['label_mask'].to_numpy().astype(np.int64)

    roc_auc = metrics.roc_auc(
        logits=eval_scores.reshape(-1, 1),
        labels=species_label_match.reshape(-1, 1),
        label_mask=label_mask.reshape(-1, 1),
        sort_descending=sort_descending,
    )[
        'macro'
    ]  # Dictionary of macro, geometric, individual & individual_var.

    average_precision = metrics.average_precision(
        eval_scores,
        species_label_match,
        label_mask=label_mask,
        sort_descending=sort_descending,
    )

    num_pos_match = sum(species_label_match == 1)
    num_neg_match = sum(species_label_match == 0)

    species_metric_eval_set.append((
        eval_species,
        average_precision,
        roc_auc,
        num_pos_match,
        num_neg_match,
        eval_set_name,
    ))

  return species_metric_eval_set


def write_results_to_csv(
    metric_results: Sequence[tuple[str, float, float, str]],
    write_results_dir: str,
    write_filename: str | None,
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
    write_filename: A specified name for the eval results file.
  """

  write_results_path = os.path.join(write_results_dir, write_filename)
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


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
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

  dot_prod = vector_a @ vector_b.T
  norm_prod = (
      np.linalg.norm(vector_a, axis=-1, keepdims=True)
      * np.linalg.norm(vector_b, axis=-1, keepdims=True).T
  )

  return dot_prod / norm_prod
