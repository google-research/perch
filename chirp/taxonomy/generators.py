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

"""CSV parsers for generating namespaces, mappings, etc."""

import csv
import dataclasses
import functools
from typing import List

from chirp import path_utils
from chirp.taxonomy import namespace

EBIRD2021_DATA_PATH = 'taxonomy/data/source_data/eBird_Taxonomy_v2021.csv'


@dataclasses.dataclass
class GeneratorOutput:
  namespaces: List[namespace.Namespace]
  mappings: List[namespace.Mapping]
  class_lists: List[namespace.ClassList]

  def union(self, other: 'GeneratorOutput'):
    return GeneratorOutput(self.namespaces + other.namespaces,
                           self.mappings + other.mappings,
                           self.class_lists + other.class_lists)


@functools.lru_cache(maxsize=1)
def load_ebird2021_dict():
  """Load the ebird2021 data in a convenient dictionary form."""
  ebird_fp = path_utils.get_absolute_epath(EBIRD2021_DATA_PATH)
  with open(ebird_fp, 'r') as f:
    dr = csv.DictReader(f)
    rows = [r for r in dr if r['CATEGORY'] == 'species']
  species_dict = {}
  for r in rows:
    if not r['SPECIES_CODE']:
      continue
    species_dict[r['SPECIES_CODE']] = {
        'species': r['SPECIES_CODE'].lower(),
        'genus': r['SCI_NAME'].split(' ')[0].lower(),
        'family': r['FAMILY'].split(' ')[0].lower(),
        'order': r['ORDER1'].lower(),
        'sci_name': r['SCI_NAME'].lower(),
    }
  return species_dict


@functools.lru_cache(maxsize=1)
def load_ebird2021_issf_dict():
  """Create the mapping from ebird subspecies codes (issf) to species."""
  # Note: 'issf' is short for 'Identifiable Sub-specific Form.'
  ebird_fp = path_utils.get_absolute_epath(EBIRD2021_DATA_PATH)
  with open(ebird_fp, 'r') as f:
    dr = csv.DictReader(f)
    rows = [r for r in dr if r['CATEGORY'] == 'issf']
  issf_dict = {}
  for r in rows:
    if not r['SPECIES_CODE']:
      continue
    issf_dict[r['SPECIES_CODE']] = r['REPORT_AS']
  return issf_dict


def generate_ebird2021():
  """Generate the ebird2021 namespace file."""
  species_dict = load_ebird2021_dict()
  issf_dict = load_ebird2021_issf_dict()

  def _to_set(key):
    return set([data[key] for data in species_dict.values()])

  species = sorted(_to_set('species'))

  # Namespaces
  ebird_namespace = namespace.Namespace('ebird2021', _to_set('species'))
  ebird_issf = namespace.Namespace('ebird2021_issf', set(issf_dict.keys()))
  ebird_genera = namespace.Namespace('bird_genera', _to_set('genus'))
  ebird_families = namespace.Namespace('bird_families', _to_set('family'))
  ebird_orders = namespace.Namespace('bird_orders', _to_set('order'))
  clements_namespace = namespace.Namespace('clements', _to_set('sci_name'))

  # Taxonomic Mappings
  issf_to_species = namespace.Mapping(
      'issf_to_ebird2021', 'ebird2021_issf', 'ebird2021',
      [(k, v) for (k, v) in sorted(issf_dict.items())])

  species_to_genus = namespace.Mapping(
      'ebird2021_to_genus',
      'ebird2021',
      'bird_genera',
      [(sp, species_dict[sp]['genus']) for sp in species],
  )

  species_to_family = namespace.Mapping(
      'ebird2021_to_family',
      'ebird2021',
      'bird_families',
      [(sp, species_dict[sp]['family']) for sp in species],
  )

  species_to_order = namespace.Mapping(
      'ebird2021_to_order',
      'ebird2021',
      'bird_orders',
      [(sp, species_dict[sp]['order']) for sp in species],
  )

  species_to_clements = namespace.Mapping(
      'ebird2021_to_clements',
      'ebird2021',
      'clements',
      [(sp, species_dict[sp]['sci_name']) for sp in species],
  )

  clements_to_species = namespace.Mapping(
      'clements_to_ebird2021',
      'clements',
      'ebird2021',
      [(species_dict[sp]['sci_name'], sp) for sp in species],
  )

  return GeneratorOutput([
      clements_namespace, ebird_namespace, ebird_genera, ebird_families,
      ebird_orders, ebird_issf
  ], [
      species_to_genus, species_to_family, species_to_order,
      clements_to_species, species_to_clements, issf_to_species
  ], [])


def generate_ioc_12_2_to_ebird2021(
    ioc_to_clements: namespace.Mapping,
    clements_to_ebird: namespace.Mapping) -> namespace.Mapping:
  """Generate the IOC to ebird mapping."""
  ioc_to_clements_dict = ioc_to_clements.to_dict()
  clements_to_ebird_dict = clements_to_ebird.to_dict()
  composed = {}
  for k, v in ioc_to_clements_dict.items():
    composed[k] = clements_to_ebird_dict.get(v, 'unknown')
  return namespace.Mapping.from_dict('ioc_12_2_to_ebird2021', 'ioc_12_2',
                                     'ebird2021', composed)
