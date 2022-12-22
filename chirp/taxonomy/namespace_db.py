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

from chirp import path_utils
from chirp.taxonomy import generators
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
  namespaces: dict[str, namespace.Namespace]
  mappings: dict[str, namespace.Mapping]
  class_lists: dict[str, namespace.ClassList]

  def __repr__(self):
    return 'NamespaceDatabase'

  @classmethod
  def load_csvs(cls) -> 'NamespaceDatabase':
    """Load the database from CSVs."""
    namespaces = {}
    mappings = {}
    class_lists = {}

    # Load generated ebird data.
    generated_data = generators.generate_ebird2021()
    for ns in generated_data.namespaces:
      namespaces[ns.name] = ns
    for mapping in generated_data.mappings:
      mappings[mapping.name] = mapping
    for class_list in generated_data.class_lists:
      class_lists[class_list.name] = class_list

    # Load CSV data.
    namespace_csvs = path_utils.listdir(NAMESPACES_PATH)
    namespace_csvs = [p for p in namespace_csvs if p.endswith('.csv')]
    for c in namespace_csvs:
      filepath = path_utils.get_absolute_epath(os.path.join(NAMESPACES_PATH, c))
      with open(filepath, 'r') as f:
        space = namespace.Namespace.from_csv(f)
        namespaces[space.name] = space

    mappings_csvs = path_utils.listdir(MAPPINGS_PATH)
    mappings_csvs = [p for p in mappings_csvs if p.endswith('.csv')]
    for c in mappings_csvs:
      filepath = path_utils.get_absolute_epath(os.path.join(MAPPINGS_PATH, c))
      with open(filepath, 'r') as f:
        mapping = namespace.Mapping.from_csv(c[:-4], f)
        mappings[mapping.name] = mapping

    class_list_csvs = path_utils.listdir(CLASS_LISTS_PATH)
    class_list_csvs = [p for p in class_list_csvs if p.endswith('.csv')]
    for c in class_list_csvs:
      filepath = path_utils.get_absolute_epath(
          os.path.join(CLASS_LISTS_PATH, c))
      with open(filepath, 'r') as f:
        class_list = namespace.ClassList.from_csv(c[:-4], f)
        class_lists[class_list.name] = class_list

    # Create a ClassList for each namespace.
    for space in namespaces.values():
      class_lists[space.name] = space.to_class_list()

    # Add the IOC->ebird2021 mapping.
    ioc_map = generators.generate_ioc_12_2_to_ebird2021(
        mappings['ioc_12_2_to_clements'],
        mappings['clements_to_ebird2021'],
    )
    mappings[ioc_map.name] = ioc_map

    # Check consistency.
    for space in namespaces.values():
      if space.name in namespaces and space != namespaces[space.name]:
        raise ValueError(f'Multiple definitions for namespace {space.name}')
    for mapping in mappings.values():
      if mapping.source_namespace not in namespaces:
        raise ValueError(f'Mapping {mapping.name} has an unknown source '
                         f'namespace {mapping.source_namespace}.')
      if mapping.target_namespace not in namespaces:
        raise ValueError(f'Mapping {mapping.name} has an unknown target '
                         f'namespace {mapping.target_namespace}.')
    for class_list in class_lists.values():
      if class_list.namespace not in namespaces:
        raise ValueError(f'ClassList {class_list.name} has an unknown '
                         f'namespace {class_list.namespace}.')

    db = NamespaceDatabase(namespaces, mappings, class_lists)
    return db
