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

"""Utilities to filter/scrub data."""
import enum
import functools
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Union

from chirp.data import sampling_utils as su
import numpy as np
import pandas as pd


class MaskOp(enum.Enum):
  """Operations used for selecting samples.

  Takes as input a dataframe and returns boolean pd.Series corresponding to the
  selected samples.
  """
  NOT_IN = 'not_in'
  IN = 'in'


class TransformOp(enum.Enum):
  """Operations that transform the dataFrame.

  Take as input a dataframe, and return an updated version of this dataframe.
  """
  SCRUB = 'scrub'
  SCRUB_ALL_BUT = 'scrub_all_but'
  FILTER = 'filter'
  SAMPLE_UNDER_CONSTRAINTS = 'sample_under_constraints'


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


def is_in(feature_dict: Dict[str, Any], key: str,
          values: List[SerializableType]) -> bool:
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
    TransformOp.SAMPLE_UNDER_CONSTRAINTS:
        su.sample_recordings_under_constraints,
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
