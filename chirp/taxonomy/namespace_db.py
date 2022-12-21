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

"""Database of bioacoustic label domains."""

import dataclasses
import functools
import os
from typing import Dict

from chirp import path_utils
from chirp.taxonomy import namespace

NAMESPACES_PATH = 'taxonomy/data/namespaces'
MAPPINGS_PATH = 'taxonomy/data/mappings'
CLASS_LISTS_PATH = 'taxonomy/data/class_lists'


@functools.lru_cache(maxsize=1)
def load_db() -> 'NamespaceDatabase':
  """Loads the NamespaceDatabase and caches the result.

  The cache can be cleared with 'namespace_db.load_db.clear_cache()', which
  may be helpful when adding new data in a colab session.

  Returns:
    A NamespaceDatabase.
  """
  return NamespaceDatabase.load_csvs()


@dataclasses.dataclass
class NamespaceDatabase:
  """A database of Namespaces, Mappings, and ClassLists."""
  namespaces: Dict[str, namespace.Namespace]
  mappings: Dict[str, namespace.Mapping]
  class_lists: Dict[str, namespace.ClassList]

  def __repr__(self):
    return 'NamespaceDatabase'

  @classmethod
  def load_csvs(cls) -> 'NamespaceDatabase':
    """Load the database from CSVs."""
    namespace_csvs = path_utils.listdir(NAMESPACES_PATH)
    namespace_csvs = [p for p in namespace_csvs if p.endswith('.csv')]
    namespaces = {}
    for c in namespace_csvs:
      filepath = path_utils.get_absolute_epath(os.path.join(NAMESPACES_PATH, c))
      with open(filepath, 'r') as f:
        space = namespace.Namespace.from_csv(f)
        namespaces[space.name] = space

    mappings_csvs = path_utils.listdir(MAPPINGS_PATH)
    mappings_csvs = [p for p in mappings_csvs if p.endswith('.csv')]
    mappings = {}
    for c in mappings_csvs:
      filepath = path_utils.get_absolute_epath(os.path.join(MAPPINGS_PATH, c))
      with open(filepath, 'r') as f:
        mapping = namespace.Mapping.from_csv(c[:-4], f)
        if mapping.source_namespace not in namespaces:
          raise ValueError(f'Mapping {c} has an unknown source namespace '
                           '{mapping.source_namespace}.')
        if mapping.target_namespace not in namespaces:
          raise ValueError(f'Mapping {c} has an unknown target namespace '
                           '{mapping.target_namespace}.')

        mappings[mapping.name] = mapping

    class_list_csvs = path_utils.listdir(CLASS_LISTS_PATH)
    class_list_csvs = [p for p in class_list_csvs if p.endswith('.csv')]
    class_lists = {}
    for c in class_list_csvs:
      filepath = path_utils.get_absolute_epath(
          os.path.join(CLASS_LISTS_PATH, c))
      with open(filepath, 'r') as f:
        class_list = namespace.ClassList.from_csv(c[:-4], f)
        if class_list.namespace not in namespaces:
          raise ValueError(
              f'ClassList {c} has an unknown namespace {class_list.namespace}.')
        class_lists[class_list.name] = class_list

    db = NamespaceDatabase(namespaces, mappings, class_lists)
    db._populate_natural_class_lists()
    return db

  def _populate_natural_class_lists(self):
    """Create ClassLists corresponding to each namespace."""
    for d in self.namespaces.values():
      if d.name not in self.class_lists:
        self.class_lists[d.name] = d.to_class_list()
      else:
        raise ValueError(
            f'Tried to add the natural class set for domain {d.name}, '
            'but found a class set with the same name.')
