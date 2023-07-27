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

"""Tools for handling namespaces of classes."""
from __future__ import annotations

import csv
import dataclasses
import io
from typing import Iterable

from jax import numpy as jnp
import tensorflow as tf

UNKNOWN_LABEL = "unknown"


@dataclasses.dataclass
class Namespace:
  """A namespace is simply a set of labels.

  Note that unknown labels cannot be in a namespace.

  Attributes:
    classes: A frozenset of labels.
  """

  classes: frozenset[str]

  def __post_init__(self):
    if UNKNOWN_LABEL in self.classes:
      raise ValueError("unknown class")


@dataclasses.dataclass
class Mapping:
  """A mapping maps labels from one namespace to labels in another.

  Note that this is a n:1 mapping, i.e., multiple labels in the source namespace
  can map to the same label in the target namespace.

  The source and target namespace are referred to by their name. This name must
  be resolved using the taxonomy database.

  Note that labels cannot be mapped to unknown. Instead, these labels should be
  simply excluded from the mapping. The end-user is responsible for deciding
  whether to map missing keys to unknown or whether to raise an error, e.g.,
  by using:

    mapping.mapped_pairs.get(source_label, namespace.UNKNOWN_LABEL)

  Attributes:
    source_namespace: The name of the source namespace.
    target_namespace: The name of the target namespace.
    mapped_pairs: The mapping from labels in the source namespace to labels in
      the target namespace.
  """

  source_namespace: str
  target_namespace: str
  mapped_pairs: dict[str, str]

  def __post_init__(self):
    if UNKNOWN_LABEL in self.mapped_pairs.values():
      raise ValueError("unknown target class")


@dataclasses.dataclass
class ClassList:
  """A list of labels.

  A class list is a list of labels in a particular order, e.g., to reflect the
  output of a model.

  Class lists can contain the unknown label. All other labels must belong to a
  namespace.

  Class lists cannot contain duplicate entries.

  Attributes:
    namespace: The name of the namespace these class labels belong to.
    classes: The list of classes.
  """

  namespace: str
  classes: tuple[str, ...]

  def __post_init__(self):
    if len(set(self.classes)) != len(self.classes):
      raise ValueError("duplicate entries in class list")

  @classmethod
  def from_csv(cls, csv_data: Iterable[str]) -> "ClassList":
    """Parse a class list from a CSV file.

    The file must contain the namespace in the first column of the first row.
    The first column of the remaining rows are assumed to contain the classes.

    Args:
      csv_data: Any iterable which can be passed on to `csv.reader`.

    Returns:
      The parsed class list.
    """
    reader = csv.reader(csv_data)
    namespace = next(reader)[0]
    classes = tuple(row[0].strip() for row in reader)
    return ClassList(namespace, classes)

  def to_csv(self) -> str:
    """Write a class list to a CSV file.

    See `from_csv` for a description of the file format.

    It can be useful to write the class lists to disk so that the model can be
    loaded correctly, even if class lists change. However, note that in this
    case none of the mappings are guaranteed to still work.

    Returns:
      A string containing the namespace and the class labels as rows.
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(self.namespace)
    writer.writerows(self.classes)
    return buffer.getvalue()

  def get_class_map_tf_lookup(
      self, target_class_list: ClassList
  ) -> tuple[tf.lookup.StaticHashTable, tf.Tensor]:
    """Create a static hash map for class indices.

    Create a lookup table for use in TF Datasets, for, eg, converting between
    ClassList defined for a dataset to a ClassList used as model outputs.
    Classes in the source ClassList which do not appear in the target_class_list
    will be mapped to -1. It is recommended to drop these labels subsequently
    with: tf.gather(x, tf.where(x >= 0)[:, 0])

    Args:
      target_class_list: Class list to target.

    Returns:
      A tensorflow StaticHashTable and an indicator vector for the image of
      the classlist mapping.
    """
    if self.namespace != target_class_list.namespace:
      raise ValueError("namespaces must match when creating a class map.")
    intersection = set(self.classes) & set(target_class_list.classes)
    keys = [i for i, k in enumerate(self.classes) if k in intersection]
    values = [
        i for i, k in enumerate(target_class_list.classes) if k in intersection
    ]
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values, tf.int64, tf.int64),
        default_value=-1,
    )
    image_mask = tf.constant(
        [k in self.classes for k in target_class_list.classes],
        tf.int64,
    )
    return table, image_mask

  def get_namespace_map_tf_lookup(
      self, mapping: Mapping
  ) -> tf.lookup.StaticHashTable:
    """Create a tf.lookup.StaticHasTable for namespace mappings.

    Args:
      mapping: Mapping to apply.

    Returns:
      A Tensorflow StaticHashTable and the image ClassList in the mapping's
      target namespace.
    """
    target_class_list = self.apply_namespace_mapping(mapping)
    target_class_indices = {
        k: i for i, k in enumerate(target_class_list.classes)
    }
    keys = list(range(len(self.classes)))
    values = [
        target_class_indices[mapping.mapped_pairs[k]] for k in self.classes
    ]
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values, tf.int64, tf.int64),
        default_value=-1,
    )
    return table

  def apply_namespace_mapping(
      self, mapping: Mapping, keep_unknown: bool | None = None
  ) -> ClassList:
    """Apply a namespace mapping to this class list.

    Args:
      mapping: The mapping to apply.
      keep_unknown: How to handle unknowns. If true, then unknown labels in the
        class list are maintained as unknown in the mapped values. If false then
        the unknown value is discarded. The default (`None`) will raise an error
        if an unknown value is in the source classt list.

    Returns:
      A class list which is the result of applying the given mapping to this
      class list.

    Raises:
      KeyError: If a class in not the mapping, or if the class list contains
      an unknown token and `keep_unknown` was not specified.
    """
    if mapping.source_namespace != self.namespace:
      raise ValueError("mapping source namespace does not match class list's")
    mapped_pairs = mapping.mapped_pairs
    if keep_unknown:
      mapped_pairs = mapped_pairs | {UNKNOWN_LABEL: UNKNOWN_LABEL}
    return ClassList(
        mapping.target_namespace,
        tuple(
            dict.fromkeys(
                mapped_pairs[class_]
                for class_ in self.classes
                if class_ != UNKNOWN_LABEL or keep_unknown in (True, None)
            )
        ),
    )

  def get_class_map_matrix(
      self,
      target_class_list: ClassList,
      mapping: Mapping | None = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct a binary matrix for mapping to another class list.

    Args:
      target_class_list: Class list to map into.
      mapping: Namespace mapping, required if the source and target are in
        different namespaces.

    Returns:
      A binary matrix mapping self to target_class_list and an indicator vector
      for the image of the mapping.
    """
    if self.namespace != target_class_list.namespace and mapping is None:
      raise ValueError(
          "If source and target classes are from different namespaces, a"
          " namespace mapping must be provided."
      )
    elif self.namespace == target_class_list.namespace and mapping is not None:
      raise ValueError(
          "If source and target classes are the same, no mapping should be"
          " provided."
      )
    matrix = jnp.zeros([len(self.classes), len(target_class_list.classes)])

    target_idxs = {k: i for i, k in enumerate(target_class_list.classes)}
    for i, class_ in enumerate(self.classes):
      if mapping is not None:
        class_ = mapping.mapped_pairs[class_]
      if class_ in target_idxs:
        j = target_idxs[class_]
        matrix = matrix.at[i, j].set(1)
    return matrix, jnp.any(matrix, axis=0)
