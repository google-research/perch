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

"""Utilities to process data. Used to create upstream/downstream datasets."""
import enum
import functools
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Union

from jax import random
import numpy as np
import pandas as pd


class MaskOp(enum.Enum):
  """Operations used for selecting samples.

  Takes as input a dataframe and returns boolean pd.Series corresponding to the
  selected samples.
  """
  NOT_IN = 'not_in'
  IN = 'in'
  SAMPLE_N_PER_CLASS = 'sample_n_per_class'
  SAMPLE_CLASSES = 'sample_classes'


class TransformOp(enum.Enum):
  """Operations that transform the dataFrame.

  Take as input a dataframe, and return an updated version of this dataframe.
  """
  SCRUB = 'scrub'
  SCRUB_ALL_BUT = 'scrub_all_but'
  FILTER = 'filter'
  SAMPLE = 'sample'


SerializableType = Union[List[Union[int, str, bytes]], MaskOp, TransformOp]


class Query(NamedTuple):
  """The main interface for processing operations.

  A query is serializable.

  It contains an operation (op), along with its kwargs. Additionally,
  for 'masking query' (when the op is a MaskOp), a complement option can be
  activated to return the complement of what the original query would have
  returned. Combined with consistent PRNG seeding, this feature makes it easy to
  partition data for training and evaluation.
  """
  op: Union[MaskOp, TransformOp]
  kwargs: Dict[str, SerializableType]
  complement: bool = False


class QuerySequence(NamedTuple):
  """A sequence of Queries.

  Contains a sequence of Query to be applied sequentially on a dataframe.
  This sequence can be targeted to a subpopulation of samples through specifying
  a mask_query (i.e. a Query whose op is a MaskOp), for instance only
  scrubbing bg_labels from a specific subset of species.
  """
  queries: Sequence[Query]
  mask_query: Optional[Query] = None


def apply_query(
    df: pd.DataFrame,
    query: Query,
) -> Union[pd.DataFrame, pd.Series]:
  """Applies a query on a DataFrame.

  Args:
    df: The dataframe on which the query is applied.
    query: The query to apply.

  Returns:
    The new version of the dataFrame (or Series) after applying the query.
  """
  if query.complement and not isinstance(query.op, MaskOp):
    raise ValueError(
        'Taking the complement of a non masking query does not make sense.')
  updated_df = OPS[query.op](df, **query.kwargs)
  if query.complement:
    return ~updated_df
  else:
    return updated_df


def apply_sequence(
    df: pd.DataFrame,
    query_sequence: QuerySequence,
) -> Union[pd.DataFrame, pd.Series]:
  """Applies a QuerySequence to a DataFrame.

  Args:
    df: The DataFrame on which to apply the query.
    query_sequence: The QuerySequence to apply to df.

  Returns:
    The updated version of the df, where all the queries in
    query_sequence.queries
    have been sequentially applied in the specified order.
  """
  if query_sequence.mask_query is not None:
    mask = apply_query(df, query_sequence.mask_query)
    assert mask.dtype == bool
    modifiable_df = df[mask]
    frozen_df = df[~mask]
    for query in query_sequence.queries:
      modifiable_df = apply_query(modifiable_df, query)
    return pd.concat([frozen_df, modifiable_df])
  else:
    for query in query_sequence.queries:
      df = apply_query(df, query)
    return df


def sample_groups(df: pd.DataFrame, n_groups: int, group_key: str,
                  seed: random.PRNGKeyArray) -> pd.Series:
  """Samples n_groups distinct groups of samples from the dataframe.

  A group is defined by group_key. For instance, if group_key='species_code',
  then the function will sample n_groups distinct species from all classes
  currently present in df.

  Args:
    df: The df to samples groups from.
    n_groups: The number of groups to randomly sample from df.
    group_key: The name of the field that defines each group.
    seed: The jax key used for sampling. Needed for reproducibility.

  Returns:
    A boolean pd.Series, indicating whether each sample belongs to a group that
    was sampled, or not.

  """
  all_groups = df[group_key].explode(group_key).unique()
  if len(all_groups) < n_groups:
    raise ValueError(
        f'Cannot sample {n_groups} groups from {len(all_groups)} total groups')
  # Cannot call random.choice directly on all_groups, jax only support numeric
  # types
  sampled_groups = all_groups[np.array(
      random.choice(
          seed, np.arange(len(all_groups)), shape=(n_groups,), replace=False))]

  def test_in(value, groups=sampled_groups):
    return np.any([group in value for group in groups])

  mask = df[group_key].map(test_in)
  return mask


def sample_recordings_per_group(df: pd.DataFrame,
                                samples_per_group: int,
                                group_key: str,
                                seed: random.PRNGKeyArray,
                                allow_overlap: bool = True) -> pd.Series:
  """Samples some provided number of recordings from each group.

  C.f. doc of sample_groups() for more info on the meaning of 'group'.

  Args:
    df: The df to sample from.
    samples_per_group: The number of recordings to sample for each group.
    group_key: The name of the field that defines each group.
    seed: The jax key used for sampling. Needed for reproducibility.
    allow_overlap: A sample may belong to multiple groups, as would be the case
      if group_key='species_code' for instance. Therefore, the total number of
      samples returned may be lower than samples_per_group * total_nb_groups,
      unless allow_overlap is set to False.

  Returns:
    Note that a sample may belong to multiple groups, as
      would be the case if group_key='species_code'.

  """

  final_mask = pd.Series(np.zeros(len(df))).astype(bool)
  unique_groups = df[group_key].explode(group_key).unique()
  for group in unique_groups:

    def test_in(value, group=group):
      return group in value

    not_yet_chosen = ~final_mask
    if allow_overlap:
      # Below, we may be re-sampling recordings that were previously sampled
      # by another group.
      candidates = df[group_key].map(test_in)
    else:
      candidates = not_yet_chosen * df[group_key].map(test_in)
    candidates_indexes = np.where(candidates)[0]
    available_samples = len(candidates_indexes)
    if available_samples < samples_per_group:
      raise ValueError(
          f'Cannot sample {samples_per_group} recordings when there are only {available_samples} recordings for group {group}'
      )
    sampled_recording_indexes = np.array(
        random.choice(
            seed, candidates_indexes, shape=(samples_per_group,),
            replace=False))
    group_mask = np.zeros(len(df)).astype(bool)
    group_mask[sampled_recording_indexes] = True
    final_mask += group_mask
  return final_mask


def is_in(feature_dict: Dict[str, Any], key: str, values: List[Any]) -> bool:
  """Ensures if feature_dict[key] is in values.

  Useful for filtering.

  Args:
    feature_dict: A dictionary that represents the row (=recording) to be
      potentially filtered in a DataFrame.
    key: The field from feature_dict used for filtering.
    values: The set of values that feature_dict[key] needs to be belong to in
      order to trigger a True response.

  Returns:
    True if feature_dict[key] is in values, False otherwise.
  """
  if key not in feature_dict:
    raise ValueError(f'{key} is not a correct field. Please choose among'
                     f'{list(feature_dict.keys())}')
  expected_type = type(feature_dict[key])
  for index, val in enumerate(values):
    if not isinstance(val, expected_type):
      raise TypeError(
          'Values[{}] has type {}, while feature_dict[{}] has type {}'.format(
              index, type(val), key, expected_type))
  return feature_dict[key] in values


def is_not_in(feature_dict: Dict[str, Any], key: str,
              values: List[SerializableType]) -> bool:
  return not is_in(feature_dict, key, values)


def scrub(feature_dict: Dict[str, Any],
          key: str,
          values: Sequence[SerializableType],
          all_but: bool = False) -> Dict[str, Any]:
  """Removes any occurence of any value in values from feature_dict[key].

  Args:
    feature_dict: A dictionary that represents the row (=recording) to be
      potentially scrubbed in a DataFrame.
    key: The field from feature_dict used for scrubbing.
    values: The values that will be scrubbed from feature_dict[key].
    all_but: If activated, will scrub every value, except those specified.

  Returns:
    A copy of feature_dict, where all values at key have been scrubbed.
  """

  if key not in feature_dict:
    raise ValueError(f'{key} is not a correct field.'
                     f'Please choose among {list(feature_dict.keys())}')
  if type(feature_dict[key]) not in [list, np.ndarray]:
    raise TypeError('Can only scrub values from lists/ndarrays. Current column'
                    'is of type {}'.format(type(feature_dict[key])))
  # Using this 'dirty' syntax because values and feature_dict[key] could be
  # list or ndarray -> using the 'not values' to check emptiness does not work.
  if len(values) == 0 or len(feature_dict[key]) == 0:
    return feature_dict
  data_type = type(feature_dict[key][0])
  for index, val in enumerate(values):
    if not isinstance(val, data_type):
      raise TypeError(
          'Values[{}] has type {}, while values in feature_dict[{}] have type {}'
          .format(index, type(val), key, data_type))
  # Avoid changing the feature_dict in-place.
  new_feature_dict = feature_dict.copy()
  if all_but:
    new_feature_dict[key] = [x for x in feature_dict[key] if x in values]
  else:
    new_feature_dict[key] = [x for x in feature_dict[key] if x not in values]
  return new_feature_dict


def filter_df(df: pd.DataFrame, mask_op: MaskOp,
              op_kwargs: Dict[str, SerializableType]):
  """Filters a dataframe based on the output of the mask_op.

  Args:
    df: The dataframe to be filtered.
    mask_op: The operation that generates the binary mask used for filtering.
    op_kwargs: kwargs to be passed to the mask_op.

  Returns:
    The filtered dataframe
  """
  mask_query = Query(op=mask_op, kwargs=op_kwargs, complement=False)
  return df[apply_query(df, mask_query)]


OPS = {
    MaskOp.IN:
        lambda df, **kwargs: df.apply(
            functools.partial(is_in, **kwargs), axis=1, result_type='expand'),
    MaskOp.NOT_IN:
        lambda df, **kwargs: df.apply(
            functools.partial(is_not_in, **kwargs),
            axis=1,
            result_type='expand'),
    MaskOp.SAMPLE_CLASSES:
        functools.partial(sample_groups, group_key='species_code'),
    MaskOp.SAMPLE_N_PER_CLASS:
        functools.partial(
            sample_recordings_per_group,
            group_key='species_code',
            allow_overlap=True),
    TransformOp.SCRUB:
        lambda df, **kwargs: df.apply(
            functools.partial(scrub, **kwargs), axis=1, result_type='expand'),
    TransformOp.SCRUB_ALL_BUT:
        lambda df, **kwargs: df.apply(
            functools.partial(functools.partial(scrub, all_but=True), **kwargs),
            axis=1,
            result_type='expand'),
    TransformOp.FILTER:
        filter_df
}
