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

"""Database of bioacoustic label domains."""

import dataclasses
import functools
import os
from typing import Sequence, Tuple

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
          os.path.join(CLASS_LISTS_PATH, c)
      )
      with open(filepath, 'r') as f:
        class_list = namespace.ClassList.from_csv(c[:-4], f)
        class_lists[class_list.name] = class_list

    # Create a ClassList for each namespace.
    for space in namespaces.values():
      class_lists[space.name] = space.to_class_list()

    # Check consistency.
    for space in namespaces.values():
      if space.name in namespaces and space != namespaces[space.name]:
        raise ValueError(f'Multiple definitions for namespace {space.name}')
    for mapping in mappings.values():
      if mapping.source_namespace not in namespaces:
        raise ValueError(
            f'Mapping {mapping.name} has an unknown source '
            f'namespace {mapping.source_namespace}.'
        )
      if mapping.target_namespace not in namespaces:
        raise ValueError(
            f'Mapping {mapping.name} has an unknown target '
            f'namespace {mapping.target_namespace}.'
        )
    for class_list in class_lists.values():
      if class_list.namespace not in namespaces:
        raise ValueError(
            f'ClassList {class_list.name} has an unknown '
            f'namespace {class_list.namespace}.'
        )

    db = NamespaceDatabase(namespaces, mappings, class_lists)
    return db

  def generate_xenocanto_10_1_to_ebird2021(
      self, xenocanto_species: Sequence[str]
  ) -> Tuple[namespace.Mapping, Sequence[str]]:
    r"""Generates a mapping from xenocanto scientific names to ebird2021.

    As of January 2023, Xeno Canto is using the IOC 10.1 taxonomy. We don't have
    access to a good mapping from this to ebird2021, so we have to roll one from
    multiple sources. We have a good mapping from IOC 12.2->ebird2021.

    There are two kinds of problems which arise: Updates between IOC 10.1
    and 12.2 (described at worldbirdnames.com), and species in IOC which map to
    subspecies in Clements/ebird.

    To handle this, we first 'update' XenoCanto scientific names to IOC 12.2.
    Where needed, we convert IOC to ebird issf's, then to species. Otherwise, we
    convert directly to species. Here's a diagram:

    +----+   +------+   +-----------+
    | XC |-->| 12.2 |-->| ebird2021 |
    +----+   +------+   +-----------+

    Args:
      xenocanto_species: A sequence of species from Xeno-Canto.

    Returns:
      Mapping for provided XC Species to ebird2021 codes, and a list of species
      which could not be mapped successfully.
    """
    xc_updates = self.mappings['xenocanto_to_ioc_12_2'].to_dict()
    ioc_to_ebird = self.mappings['ioc_12_2_to_ebird2021'].to_dict()
    composite = {}
    misses = []

    for sp in xenocanto_species:
      sp = sp.lower()
      sp = xc_updates.get(sp, sp)
      if sp in ioc_to_ebird:
        composite[sp] = ioc_to_ebird[sp]
      else:
        misses.append(sp)
    composite = namespace.Mapping.from_dict(
        'xc_to_ebird2021', 'xenocanto_10_1', 'ebird2021', composite
    )
    return composite, misses
