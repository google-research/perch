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

"""Tools for handling namespaces of classes."""

import csv
import dataclasses
from typing import Dict, Iterable, Sequence, Set

from jax import numpy as jnp
import tensorflow as tf

UNKNOWN_LABEL = 'unknown'


@dataclasses.dataclass
class Namespace:
  """An unordered collection of classes."""

  name: str
  classes: Set[str]

  def __contains__(self, other) -> bool:
    return other == UNKNOWN_LABEL or other in self.classes

  def __repr__(self) -> str:
    return f'Namespace_{self.name}'

  @property
  def size(self) -> int:
    return len(self.classes)

  def __eq__(self, other) -> bool:
    if not isinstance(other, Namespace):
      return False
    return sorted(self.classes) == sorted(other.classes)

  @classmethod
  def from_csv(cls, csv_data: Iterable[str]) -> 'Namespace':
    reader = csv.DictReader(csv_data)
    name = reader.fieldnames[0]
    classes = []
    for row in reader:
      classes.append(row[name].strip())
    return Namespace(name, set(classes))

  def to_class_list(self) -> 'ClassList':
    return ClassList(self.name, self.name, sorted(self.classes))


@dataclasses.dataclass
class Mapping:
  """A mapping between two Namespaces."""

  name: str
  source_namespace: str
  target_namespace: str
  mapped_pairs: Sequence[tuple[str, str]]

  def __repr__(self) -> str:
    return (
        f'Mapping {self.name} from {self.source_namespace} '
        f'to {self.target_namespace}'
    )

  def __eq__(self, other) -> bool:
    if not isinstance(other, Mapping):
      return False
    elif self.source_namespace != other.source_namespace:
      return False
    elif self.target_namespace != other.target_namespace:
      return False
    elif sorted(self.mapped_pairs) != sorted(other.mapped_pairs):
      return False
    if len(self.mapped_pairs) != len(other.mapped_pairs):
      return False
    for pair_self, pair_other in zip(
        sorted(self.mapped_pairs), sorted(other.mapped_pairs)
    ):
      if pair_self[0] != pair_other[0] or pair_self[1] != pair_other[1]:
        return False
    return True

  @classmethod
  def from_dict(
      cls,
      name: str,
      source_namespace: str,
      target_namespace: str,
      mapped_pairs: Dict[str, str],
  ) -> 'Mapping':
    pairs = tuple((k, v) for (k, v) in mapped_pairs.items())
    return Mapping(name, source_namespace, target_namespace, pairs)

  @classmethod
  def from_csv(cls, name: str, csv_data: Iterable[str]) -> 'Mapping':
    reader = csv.DictReader(csv_data)
    source_namespace = reader.fieldnames[0]
    target_namespace = reader.fieldnames[1]
    pairs = []
    for row in reader:
      pairs.append((row[source_namespace], row[target_namespace]))
    return Mapping(
        name, source_namespace.strip(), target_namespace.strip(), pairs
    )

  def to_dict(self) -> dict[str, str]:
    return {m[0]: m[1] for m in self.mapped_pairs}

  def to_tf_lookup(self) -> tf.lookup.StaticHashTable:
    keys = [m[0] for m in self.mapped_pairs]
    values = [m[1] for m in self.mapped_pairs]
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=UNKNOWN_LABEL,
    )
    return table


@dataclasses.dataclass
class ClassList:
  """An ordered set of classes from a specific Domain."""

  name: str
  namespace: str
  classes: Sequence[str]

  def __contains__(self, other) -> bool:
    return other == UNKNOWN_LABEL or other in self.classes

  def __repr__(self) -> str:
    return f'ClassList {self.name} in {self.namespace}'

  def __eq__(self, other):
    if not isinstance(other, ClassList):
      return False
    return sorted(self.classes) == sorted(other.classes)

  @classmethod
  def from_csv(cls, name: str, csv_data: Iterable[str]) -> 'ClassList':
    """Parse a ClassList from a CSV."""
    reader = csv.DictReader(csv_data)
    namespace = reader.fieldnames[0].strip()
    classes = []
    for row in reader:
      classes.append(row[namespace].strip())
    return ClassList(name, namespace, classes)

  def to_csv(self) -> str:
    """Serialize to CSV string."""
    output_rows = [f'{self.namespace},comment']
    for cl in self.classes:
      output_rows.append(f'{cl},')
    return '\n'.join(output_rows) + '\n'

  @property
  def size(self) -> int:
    return len(self.classes)

  def get_index_lookup(self) -> dict[str, int]:
    return {self.classes[i]: i for i in range(self.size)}

  def get_class_map_tf_lookup(
      self, target_class_list: 'ClassList'
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
      raise ValueError('Domains must match when creating a class map.')
    intersection = [k for k in self.classes if k in target_class_list.classes]
    source_idxs = self.get_index_lookup()
    target_idxs = target_class_list.get_index_lookup()
    keys = [source_idxs[k] for k in intersection]
    values = [target_idxs[k] for k in intersection]
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys, values, tf.int64, tf.int64
    )
    table = tf.lookup.StaticHashTable(
        initializer,
        default_value=-1,
    )
    image_mask = [k in source_idxs for k in target_class_list.classes]
    image_mask = tf.constant(image_mask, tf.int64)
    return table, image_mask

  def get_namespace_map_tf_lookup(
      self, mapping: Mapping
  ) -> tuple[tf.lookup.StaticHashTable, 'ClassList']:
    """Create a tf.lookup.StaticHasTable for namespace mappings.

    Args:
      mapping: Mapping to apply.

    Returns:
      A Tensorflow StaticHashTable and the image ClassList in the mapping's
      target namespace.
    """
    mapping_dict = mapping.to_dict()
    target_class_list = self.apply_namespace_mapping(mapping)
    target_class_idxs = target_class_list.get_index_lookup()
    keys = []
    values = []
    for i, cl in enumerate(self.classes):
      if cl in mapping_dict:
        target_cl = mapping_dict[cl]
        keys.append(i)
        values.append(target_class_idxs[target_cl])
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys, values, tf.int64, tf.int64
    )
    table = tf.lookup.StaticHashTable(
        initializer=initializer,
        default_value=-1,
    )
    return table, target_class_list

  def apply_namespace_mapping(
      self, mapping: Mapping, mapped_name: str | None = None
  ) -> 'ClassList':
    """Produces a new ClassList by applying a Mapping.

    The output ClassList is in alphabetical order, and includes only the
    elements of the target Domain in the image of the source ClassList.
    (ie, sorted(D(self.classes)).)

    Args:
      mapping: The Mapping to apply.
      mapped_name: Name for output ClassList.

    Returns:
      A new ClassList in the Mapping's target namespace.
    """
    if mapped_name is None:
      mapped_name = self.name + '_' + mapping.target_namespace
    mapping_dict = mapping.to_dict()
    mapped_classes = sorted(
        set([mapping_dict[cl] for cl in self.classes if cl in mapping_dict])
    )
    return ClassList(mapped_name, mapping.target_namespace, mapped_classes)

  def get_class_map_matrix(
      self, target_class_list: 'ClassList', mapping: Mapping | None = None
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct a binary matrix for mapping to another ClassList.

    Args:
      target_class_list: ClassList to map into.
      mapping: Namespace mapping, required if the source and target are in
        different namespaces.

    Returns:
      A binary matrix mapping self to target_class_list and an indicator vector
      for the image of the mapping.
    """
    if self.namespace != target_class_list.namespace and mapping is None:
      raise ValueError(
          'If source and target classes are from different '
          'namespaces, a namespace mapping must be provided.'
      )
    elif mapping is not None:
      mapping_dict = mapping.to_dict()
    else:
      mapping_dict = {}
    matrix = jnp.zeros([self.size, target_class_list.size])
    image_mask = jnp.zeros([target_class_list.size])

    target_idxs = target_class_list.get_index_lookup()
    for i, cl in enumerate(self.classes):
      if mapping is not None and cl in mapping_dict:
        # Consider the class as a member of the target namespace.
        cl = mapping_dict[cl]
      elif mapping is not None:
        # Source class does not exist in the target namespace, so ignore it.
        continue

      if cl in target_class_list.classes:
        j = target_idxs[cl]
        matrix = matrix.at[i, j].set(1)
        image_mask = image_mask.at[j].set(1)
    return matrix, image_mask
