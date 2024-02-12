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

"""Utilities to filter/scrub data."""
import enum
import functools
from typing import Any, Dict, NamedTuple, Sequence, Union

from chirp.data import sampling_utils as su
from chirp.taxonomy import namespace_db
import numpy as np
import pandas as pd


class MergeStrategy(enum.Enum):
  """Strategy used to merge the results of parallel queries in QueryParallel."""

  OR = 'or'
  AND = 'and'
  CONCAT_NO_DUPLICATES = 'concat_no_duplicates'


class MaskOp(enum.Enum):
  """Operations used for selecting samples.

  Takes as input a dataframe and returns boolean pd.Series corresponding to the
  selected samples.
  """

  NOT_IN = 'not_in'
  CONTAINS_NO = 'contains_no'
  CONTAINS_ANY = 'contains_any'
  IN = 'in'


class TransformOp(enum.Enum):
  """Operations that transform the dataFrame.

  Take as input a dataframe, and return an updated version of this dataframe.
  """

  SCRUB = 'scrub'
  SCRUB_ALL_BUT = 'scrub_all_but'
  FILTER = 'filter'
  SAMPLE = 'sample'
  APPEND = 'append'


SerializableType = list[int | str | bytes] | MaskOp | TransformOp | Dict


class Query(NamedTuple):
  """The main interface for processing operations.

  A query is serializable.

  It contains an operation (op), along with its kwargs. Additionally,
  for 'masking query' (when the op is a MaskOp), a complement option can be
  activated to return the complement of what the original query would have
  returned. Combined with consistent PRNG seeding, this feature makes it easy to
  partition data for training and evaluation.
  """

  op: MaskOp | TransformOp
  kwargs: dict[str, SerializableType]


class QuerySequence(NamedTuple):
  """A sequence of Queries to be applied sequentially.

  Contains a sequence of Query to be applied sequentially on a dataframe.
  This sequence can be targeted to a subpopulation of samples through specifying
  a mask_query (i.e. a Query whose op is a MaskOp), for instance only
  scrubbing bg_labels from a specific subset of species.
  """

  queries: Sequence[Union[Query, 'QuerySequence', 'QueryParallel']]
  mask_query: Union[Query, 'QueryParallel'] | None = None


class QueryParallel(NamedTuple):
  """A sequence of Queries to be applied in parallel.

  Contains a sequence of Query to be applied in parallel from a given dataframe.
  Once all queries have been independently executed, we merge the resulting df
  using the merge_strategy defined.
  """

  queries: Sequence[Union[Query, QuerySequence, 'QueryParallel']]
  merge_strategy: MergeStrategy


class QueryComplement(NamedTuple):
  """Applies the complement of a query.

  The unique_key is used to uniquely identify samples. Therefore, the values
  at that field must remain **unchanged** throughout the application of query.
  """

  query: Query | QuerySequence
  unique_key: str


def apply_complement(
    df: pd.DataFrame, query_complement: QueryComplement
) -> pd.DataFrame:
  """Applies a QueryComplement.

  If the query transforms the df into a boolean Series, we just return the
  complement of the mask. For transform operations, we compare the values
  at query_complement.unique_key of samples initially present, minus those
  remaining after the application of query_complement.query. This assumes that
  (i) values in df[query_complement.unique_key] bijectively map to recordings.
  (ii) query_complement.query **does not** modify in-place this mapping.

  Args:
    df: The dataframe to apply the QueryComplement on.
    query_complement: The QueryComplement to apply.

  Returns:
    The complemented query.

  Raises:
    ValueError: Some values in df[query_complement.unique_key] are duplicates,
      which violates condition (i) above.
  """

  updated_df = APPLY_FN[type(query_complement.query)](
      df, query_complement.query
  )
  # If the query used a MaskOp (yields a boolean Series), we return the
  # complement of this boolean Series.
  if isinstance(query_complement.query, MaskOp):
    return ~updated_df
  # For other transformations, we use the unique_key to return the complement.
  else:
    key = query_complement.unique_key
    if df[key].duplicated().any():
      raise ValueError(
          f'The values at {key} should uniquely define each'
          'recording. Currently, some recordings share a similar'
          'value.'
      )
    complement_values = set(df[key]) - set(updated_df[key])
    comp_mask = df[key].apply(lambda v: v in complement_values)
    return df[comp_mask]


def apply_query(
    df: pd.DataFrame,
    query: Query,
) -> pd.DataFrame | pd.Series:
  """Applies a query on a DataFrame.

  Args:
    df: The dataframe on which the query is applied.
    query: The query to apply.

  Returns:
    The new version of the dataFrame (or Series) after applying the query.
  """
  return OPS[query.op](df, **query.kwargs)


def apply_sequence(
    df: pd.DataFrame,
    query_sequence: QuerySequence,
) -> pd.DataFrame | pd.Series:
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
    mask = APPLY_FN[type(query_sequence.mask_query)](
        df, query_sequence.mask_query
    )
    assert mask.dtype == bool
    modifiable_df = df[mask]
    frozen_df = df[~mask]
    for query in query_sequence.queries:
      modifiable_df = APPLY_FN[type(query)](modifiable_df, query)
    return pd.concat([frozen_df, modifiable_df])
  else:
    for query in query_sequence.queries:
      df = APPLY_FN[type(query)](df, query)
    return df


def apply_parallel(
    df: pd.DataFrame,
    query_parallel: QueryParallel,
) -> pd.DataFrame | pd.Series:
  """Applies a QueryParallel to a DataFrame.

  Args:
    df: The DataFrame on which to apply the query.
    query_parallel: The QueryParallel to apply to df.

  Returns:
    The updated version of the df, where all the queries in
    query_sequence.queries
    have been sequentially applied in the specified order.
  """
  all_dfs = []
  for query in query_parallel.queries:
    all_dfs.append(APPLY_FN[type(query)](df, query))

  final_df = MERGE_FN[query_parallel.merge_strategy](all_dfs)
  return final_df


def is_in(
    df: pd.DataFrame, key: str, values: list[SerializableType]
) -> pd.Series:
  """Builds a binary mask of whether `df[key]` is in `values`.

  Useful for filtering.

  Args:
    df: The DataFrame.
    key: The column used for filtering.
    values: Values to look out for.

  Returns:
    A boolean Series representing whether `df[key]` is in `values`.

  Raises:
    ValueError: 'key' does not exist in df.
    TypeError: inconsistent types in df[key] and values.
  """
  if key not in df:
    raise ValueError(
        f'{key} is not a correct field. Please choose among{list(df.columns)}'
    )
  values_types = set(type(v) for v in values)
  df_column_types = set(df[key].map(type).unique())
  if len(values_types.union(df_column_types)) != 1:
    raise TypeError("Inconsistent types between df['{key}'] and values")
  return df[key].isin(values)


def contains_any(df: pd.DataFrame, key: str, values: list[str]) -> pd.Series:
  """Builds a binary mask of whether `df[key]` contains any of `values`.

  Args:
    df: The DataFrame. Note that `df[key]` must be a Sequence, e.g. the
      background labels.
    key: The column used for filtering.
    values: Values to look out for.

  Returns:
    A boolean Series representing whether `df[key]` contains any of `values`.

  Raises:
    ValueError: key does not exist in df.
    ValueError: inconsistent types in df[key] and values.
  """
  if key not in df:
    raise ValueError(
        f'{key} is not a correct field. Please choose among{list(df.columns)}'
    )
  values_types = set(type(v) for v in values)
  df_column_types = set().union(
      *df[key].map(lambda xs: set(type(x) for x in xs))
  )
  if len(values_types.union(df_column_types)) != 1:
    raise ValueError("Inconsistent types between df['{key}'] and values")
  return df[key].map(' '.join).str.contains('|'.join(values))


def contains_no(df: pd.DataFrame, key: str, values: list[str]) -> pd.Series:
  """Builds a binary mask of whether `df[key]` contains none of `values`.

  Args:
    df: The DataFrame. Note that `df[key]` must be a Sequence, e.g. the
      background labels.
    key: The column used for filtering.
    values: Values to look out for.

  Returns:
    A boolean Series representing whether `df[key]` contains none of `values`.

  Raises:
    ValueError: key does not exist in df.
  """
  return ~contains_any(df, key, values)


def is_not_in(
    df: pd.DataFrame, key: str, values: list[SerializableType]
) -> pd.Series:
  return ~is_in(df, key, values)


def append(df: pd.DataFrame, row: dict[str, Any]):
  if set(row.keys()) != set(df.columns):
    raise ValueError
  new_df = pd.concat([df, pd.DataFrame(pd.Series(row))], ignore_index=True)
  return new_df


def scrub(
    feature_dict: dict[str, Any],
    key: str,
    values: Sequence[SerializableType],
    all_but: bool = False,
    replace_value: SerializableType | None = None,
) -> dict[str, Any]:
  """Removes any occurence of any value in values from feature_dict[key].

  Args:
    feature_dict: A dictionary that represents the row (=recording) to be
      potentially scrubbed in a DataFrame.
    key: The field from feature_dict used for scrubbing.
    values: The values that will be scrubbed from feature_dict[key].
    all_but: If activated, will scrub every value, except those specified.
    replace_value: If specified, used as a placeholder wherever a value was
      scrubbed.

  Returns:
    A copy of feature_dict, where all values at key have been scrubbed.

  Raises:
    ValueError: 'key' does not exist in df.
    TypeError: any element of 'values' has a type different from the type at
      df[key], or feature_dict[key] is not a str, list or np.ndarray.
  """

  if key not in feature_dict:
    raise ValueError(
        f'{key} is not a correct field.'
        f'Please choose among {list(feature_dict.keys())}'
    )
  if type(feature_dict[key]) not in [list, np.ndarray, str]:
    raise TypeError(
        'Can only scrub values from str/lists/ndarrays. Current column'
        'is of type {}'.format(type(feature_dict[key]))
    )
  # Using this 'dirty' syntax because values and feature_dict[key] could be
  # list or ndarray -> using the 'not values' to check emptiness does not work.
  if len(values) == 0 or len(feature_dict[key]) == 0:  # pylint: disable=g-explicit-length-test
    return feature_dict
  field_type = type(feature_dict[key][0])
  for index, val in enumerate(values):
    if not isinstance(val, field_type):
      raise TypeError(
          'Values[{}] has type {}, while values in feature_dict[{}] have'
          ' type {}'.format(index, type(val), key, field_type)
      )
  # Avoid changing the feature_dict in-place.
  new_feature_dict = feature_dict.copy()
  key_type = type(new_feature_dict[key])
  if key_type == str:
    new_feature_dict[key] = new_feature_dict[key].split(' ')

  scrub_mask = [True if x in values else False for x in new_feature_dict[key]]
  if all_but:
    scrub_mask = [not x for x in scrub_mask]
  if replace_value is None:
    new_feature_dict[key] = [
        x for x, scrub in zip(new_feature_dict[key], scrub_mask) if not scrub
    ]
  else:
    new_feature_dict[key] = [
        x if not scrub else replace_value
        for x, scrub in zip(new_feature_dict[key], scrub_mask)
    ]
  if key_type == str:
    new_feature_dict[key] = ' '.join(new_feature_dict[key])
  return new_feature_dict


def filter_df(
    df: pd.DataFrame, mask_op: MaskOp, op_kwargs: dict[str, SerializableType]
):
  """Filters a dataframe based on the output of the mask_op.

  Args:
    df: The dataframe to be filtered.
    mask_op: The operation that generates the binary mask used for filtering.
    op_kwargs: kwargs to be passed to the mask_op.

  Returns:
    The filtered dataframe
  """
  mask_query = Query(op=mask_op, kwargs=op_kwargs)
  return df[APPLY_FN[type(mask_query)](df, mask_query)]


def or_series(series_list: list[pd.Series]) -> pd.Series:
  """Performs an OR operation on a list of boolean pd.Series.

  Args:
    series_list: List of boolean pd.Series to perform OR on.

  Returns:
    The result of s_1 or ... or s_N, for s_i in series_list.

  Raises:
    TypeError: Some series in series_list is has non boolean values.
    RuntimeError: The series's indexes in series_list don't match, potentially
     meaning that series don't describe the same recordings.
  """
  reference_indexes = series_list[0].index
  if any(
      [not series.index.equals(reference_indexes) for series in series_list]
  ):
    raise RuntimeError('OR operation expects consistent Series as input')
  if any([series.dtype != bool for series in series_list]):
    raise TypeError('OR operation expects boolean Series as input.')
  return functools.reduce(lambda s1, s2: s1.add(s2), series_list)


def and_series(series_list: list[pd.Series]) -> pd.Series:
  """Performs an AND operation on a list of boolean pd.Series.

  Args:
    series_list: List of boolean pd.Series to perform AND on.

  Returns:
    The result of s_1 and ... and s_N, for s_i in series_list.

  Raises:
    TypeError: Some series in series_list is has non boolean values.
    RuntimeError: The series's indexes in series_list don't match, potentially
     meaning that series don't describe the same recordings.
  """
  reference_indexes = series_list[0].index
  if any(
      [not series.index.equals(reference_indexes) for series in series_list]
  ):
    raise RuntimeError('AND operation expects consistent Series as input')
  if any([series.dtype != bool for series in series_list]):
    raise RuntimeError('AND operation expects boolean Series as input.')
  return functools.reduce(lambda s1, s2: s1 * s2, series_list)


def concat_no_duplicates(df_list: list[pd.DataFrame]) -> pd.DataFrame:
  """Concatenates dataframes in df_list, then removes duplicates examples.

  Args:
    df_list: The list of dataframes to concatenate.

  Returns:
    The concatenated dataframe, where potential duplicated rows have been
    dropped.

  Raises:
    RuntimeError: Some series in series_list don't share the same columns.
  """
  reference_columns = set(df_list[0].columns)
  if any([set(df.columns) != reference_columns for df in df_list]):
    raise RuntimeError(
        'Concatenation expects dataframes to share the exact '
        'same set of columns.'
    )
  concat_df = pd.concat(df_list)
  # List and np.ndarray are not hashable, therefore the method
  # .duplicated() will raise an error if any of the value is of this type.
  # Instead convert to tuples for the sake of duplicate verification.
  duplicated = concat_df.applymap(
      lambda e: tuple(e) if type(e) in [list, np.ndarray] else e
  ).duplicated()
  return concat_df[~duplicated]


def filter_in_class_list(key: str, class_list_name: str) -> Query:
  """Creates a query filtering out labels not in the target class list.

  Args:
    key: Key for labels to filter. (eg, 'label'.)
    class_list_name: Name of class list to draw labels from.

  Returns:
    Query for filtering.
  """
  db = namespace_db.load_db()
  classes = list(db.class_lists[class_list_name].classes)
  return Query(
      op=TransformOp.FILTER,
      kwargs={
          'mask_op': MaskOp.IN,
          'op_kwargs': {
              'key': key,
              'values': classes,
          },
      },
  )


def filter_not_in_class_list(key: str, class_list_name: str) -> Query:
  """Creates a query filtering out labels  in the target class list.

  Args:
    key: Key for labels to filter. (eg, 'label'.)
    class_list_name: Name of class list to draw labels from.

  Returns:
    Query for filtering.
  """
  db = namespace_db.load_db()
  classes = list(db.class_lists[class_list_name].classes)
  return Query(
      op=TransformOp.FILTER,
      kwargs={
          'mask_op': MaskOp.NOT_IN,
          'op_kwargs': {
              'key': key,
              'values': classes,
          },
      },
  )


def filter_contains_no_class_list(key: str, class_list_name: str) -> Query:
  """Creates a query filtering out labels not contains in the target class list.

  Args:
    key: The column used for filtering. (eg, 'label'.) Note that `df[key]` must
      be a Sequence
    class_list_name: Name of class list to  remove  labels from.

  Returns:
    Query for filtering.
  """
  db = namespace_db.load_db()
  classes = list(db.class_lists[class_list_name].classes)
  return Query(
      op=TransformOp.FILTER,
      kwargs={
          'mask_op': MaskOp.CONTAINS_NO,
          'op_kwargs': {
              'key': key,
              'values': classes,
          },
      },
  )


def filter_contains_any_class_list(key: str, class_list_name: str) -> Query:
  """Creates a query filtering out labels which contain any of class list.

  Args:
    key: The column used for filtering. (eg, 'label'.) Note that `df[key]` must
      be a Sequence
    class_list_name: Name of class list to  remove  labels from.

  Returns:
    Query for filtering.
  """
  db = namespace_db.load_db()
  classes = list(db.class_lists[class_list_name].classes)
  return Query(
      op=TransformOp.FILTER,
      kwargs={
          'mask_op': MaskOp.CONTAINS_ANY,
          'op_kwargs': {
              'key': key,
              'values': classes,
          },
      },
  )


def scrub_all_but_class_list(key: str, class_list_name: str) -> Query:
  """Scrub everything outside the chosen class list.

  Args:
    key: Key for labels to filter. (eg, 'label'.)
    class_list_name: Name of class list containing labels to keep.

  Returns:
    Query for scrub operation.
  """
  db = namespace_db.load_db()
  classes = list(db.class_lists[class_list_name].classes)
  return Query(
      op=TransformOp.SCRUB_ALL_BUT,
      kwargs={
          'key': key,
          'values': classes,
      },
  )


APPLY_FN = {
    Query: apply_query,
    QuerySequence: apply_sequence,
    QueryComplement: apply_complement,
    QueryParallel: apply_parallel,
}

MERGE_FN = {
    MergeStrategy.OR: or_series,
    MergeStrategy.AND: and_series,
    MergeStrategy.CONCAT_NO_DUPLICATES: concat_no_duplicates,
}

OPS = {
    # pylint: disable=g-long-lambda
    MaskOp.IN: is_in,
    MaskOp.CONTAINS_NO: contains_no,
    MaskOp.CONTAINS_ANY: contains_any,
    MaskOp.NOT_IN: is_not_in,
    TransformOp.SAMPLE: su.sample_recordings,
    TransformOp.SCRUB: lambda df, **kwargs: df.apply(
        functools.partial(scrub, **kwargs), axis=1, result_type='expand'
    ),
    TransformOp.SCRUB_ALL_BUT: lambda df, **kwargs: df.apply(
        functools.partial(functools.partial(scrub, all_but=True), **kwargs),
        axis=1,
        result_type='expand',
    ),
    TransformOp.FILTER: filter_df,
    TransformOp.APPEND: append,
}
