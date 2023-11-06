# coding=utf-8
# Copyright 2023 The Perch Authors.
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

  def __str__(self):
    example_class = sorted(list(self.classes))[0]
    return (
        f"Namespace with {len(self.classes)} classes, "
        f"including '{example_class}'"
    )


@dataclasses.dataclass
class Mapping:
  """A mapping maps labels from one namespace to labels in another.

  Note that this is a n:1 mapping, i.e., multiple labels in the source namespace
  can map to the same label in the target namespace.

  The source and target namespace are referred to by their name. This name must
  be resolved using the taxonomy database.

  Note that labels (other than unknown) cannot be mapped to unknown. Instead,
  these labels should be simply excluded from the mapping. The end-user is
  responsible for deciding whether to map missing keys to unknown or whether to
  raise an error, e.g., by using:

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

  def __str__(self):
    return f"Mapping from {self.source_namespace} to {self.target_namespace}."


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

  def __str__(self):
    example_class = sorted(list(self.classes))[0]
    return (
        f"ClassList in namespace {self.namespace} with {len(self.classes)}"
        f" classes, including '{example_class}'."
    )

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
    classes = tuple(row[0].strip() for row in reader if row)
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
    writer.writerow([self.namespace])
    for class_ in self.classes:
      writer.writerow([class_])
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
    intersection = sorted(tuple(intersection))
    keys = tuple(self.classes.index(c) for c in intersection)
    values = tuple(target_class_list.classes.index(c) for c in intersection)

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
      self,
      mapping: Mapping,
      keep_unknown: bool | None = None,
      target_class_list: ClassList | None = None,
  ) -> tf.lookup.StaticHashTable:
    """Create a tf.lookup.StaticHasTable for namespace mappings.

    Args:
      mapping: Mapping to apply.
      keep_unknown: How to handle unknowns. If true, then unknown labels in the
        class list are maintained as unknown in the mapped values. If false then
        the unknown value is discarded. The default (`None`) will raise an error
        if an unknown value is in the source classt list.
      target_class_list: Optional class list for ordering of mapping output. If
        not provided, a class list consisting of the alphabetized image set of
        the mapping will be used.

    Returns:
      A Tensorflow StaticHashTable and the image ClassList in the mapping's
      target namespace.

    Raises:
      ValueError: If 'unknown' label is in source classes and keep_unknown was
      not specified.
      ValueError: If a target class list was passed and the namespace of this
      does not match the mapping target namespace.
    """
    if UNKNOWN_LABEL in self.classes and keep_unknown is None:
      raise ValueError(
          "'unknown' found in source classes. Explicitly set keep_unknown to"
          " True or False. Alternatively, remove 'unknown' from source classes"
      )
    # If no target_class_list is passed, default to apply_namespace_mapping
    if target_class_list is None:
      target_class_list = self.apply_namespace_mapping(
          mapping, keep_unknown=keep_unknown
      )
    else:
      if target_class_list.namespace != mapping.target_namespace:
        raise ValueError(
            f"target class list namespace ({target_class_list.namespace}) "
            "does not match mapping target namespace "
            f"({mapping.target_namespace})"
        )
    # Now check if 'unknown' label present in target_class_list.classes
    keep_unknown = keep_unknown and UNKNOWN_LABEL in target_class_list.classes
    # Dict which maps classes to an index
    target_class_indices = {
        k: i for i, k in enumerate(target_class_list.classes)
    }
    # Add unknown to mapped pairs
    mapped_pairs = mapping.mapped_pairs | {UNKNOWN_LABEL: UNKNOWN_LABEL}
    # If keep unknown==False, set unknown index to -1 to discard unknowns
    if not keep_unknown:
      target_class_indices[UNKNOWN_LABEL] = -1
    # Get keys and values to be used in the lookup table
    keys = list(range(len(self.classes)))
    values = [
        target_class_indices[mapped_pairs[k]]
        for k in self.classes
    ]
    # Create the static hash table. If a key doesnt exist, set as -1.
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
