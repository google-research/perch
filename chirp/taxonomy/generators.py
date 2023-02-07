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

"""CSV and json parsers for generating namespaces, mappings, etc."""

import csv
import dataclasses
import functools
import json
from typing import List

from chirp import path_utils
from chirp.taxonomy import namespace


EBIRD2021_DATA_PATH = 'taxonomy/data/source_data/eBird_Taxonomy_v2021.csv'
AUDIOSET_DATA_PATH = 'taxonomy/data/source_data/AudioSet_ontology.json'


@dataclasses.dataclass
class GeneratorOutput:
  namespaces: List[namespace.Namespace]
  mappings: List[namespace.Mapping]
  class_lists: List[namespace.ClassList]

  def union(self, other: 'GeneratorOutput'):
    return GeneratorOutput(
        self.namespaces + other.namespaces,
        self.mappings + other.mappings,
        self.class_lists + other.class_lists,
    )


@functools.lru_cache(maxsize=1)
def load_ebird2021_dict():
  """Load the ebird2021 data in a convenient dictionary form."""
  ebird_fp = path_utils.get_absolute_epath(EBIRD2021_DATA_PATH)
  with open(ebird_fp, 'r') as f:
    dr = csv.DictReader(f)
    rows = list(dr)
  codes_dict = {}
  for r in rows:
    if not r['SPECIES_CODE']:
      continue
    codes_dict[r['SPECIES_CODE']] = {
        'species': r['SPECIES_CODE'].lower(),
        # Note that this genus extraction works for almost all tag categories.
        'genus': r['SCI_NAME'].split(' ')[0].lower(),
        'family': r['FAMILY'].split(' ')[0].lower(),
        'order': r['ORDER1'].lower(),
        'sci_name': r['SCI_NAME'].lower(),
        'category': r['CATEGORY'].lower(),
        'report_as': r['REPORT_AS'].lower(),
    }
    # Apply a correction for 'spuh' tags rolling up to higher taxonomic levels.
    if r['CATEGORY'].lower() == 'spuh':
      spuh_rollup = r['SCI_NAME'].split(' ')[0].lower()
      spuh_data = codes_dict[r['SPECIES_CODE']]
      if (
          spuh_rollup == spuh_data['order']
          or spuh_rollup == spuh_data['family']
      ):
        spuh_data['genus'] = ''

  return codes_dict


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
  codes_dict = load_ebird2021_dict()

  def _to_set(key):
    return set([data[key] for data in codes_dict.values() if data[key]])

  species = sorted(
      [k for (k, v) in codes_dict.items() if v['category'] == 'species']
  )

  # Namespaces
  ebird_all = namespace.Namespace('ebird2021', set(codes_dict.keys()))
  ebird_species = namespace.Namespace('ebird2021_species', set(species))
  ebird_genera = namespace.Namespace('bird_genera', _to_set('genus'))
  ebird_families = namespace.Namespace('bird_families', _to_set('family'))
  ebird_orders = namespace.Namespace('bird_orders', _to_set('order'))
  clements_namespace = namespace.Namespace('clements', _to_set('sci_name'))

  issfs = set([k for (k, v) in codes_dict.items() if v['category'] == 'issf'])
  ebird_issf = namespace.Namespace('ebird2021_issf', issfs)

  # Taxonomic Mappings
  ebird_all_to_species = {}
  for k, v in codes_dict.items():
    if v['category'] == 'species':
      ebird_all_to_species[k] = k
    elif v['report_as'] and codes_dict[v['report_as']]['category'] == 'species':
      ebird_all_to_species[k] = v['report_as']
  # All others (hybrids, forms, spuhs, etc) don't have a mapping to species.
  # In theory we could do something for hybrids, as the scientific name
  # seems parseable in the taxonomy file (of the form: 'genus sp1 x sp2')
  # but it'll take a lot of work.
  ebird_all_to_species = namespace.Mapping.from_dict(
      'ebird2021_to_species',
      'ebird2021',
      'ebird2021_species',
      ebird_all_to_species,
  )

  def get_ebird_all_taxon_mapping(key, target_namespace):
    mapping_dict = {}
    for k, v in codes_dict.items():
      if v[key] in target_namespace.classes:
        mapping_dict[k] = v[key]
      elif not v[key]:
        # Some 'spuh' classes roll up to Order and thus have no family.
        mapping_dict[k] = 'unknown'
    mapping = namespace.Mapping.from_dict(
        f'ebird2021_to_{key}', 'ebird2021', target_namespace.name, mapping_dict
    )
    return mapping

  ebird_all_to_genus = get_ebird_all_taxon_mapping('genus', ebird_genera)
  ebird_all_to_family = get_ebird_all_taxon_mapping('family', ebird_families)
  ebird_all_to_order = get_ebird_all_taxon_mapping('order', ebird_orders)

  issf_dict = {
      k: v['report_as']
      for k, v in codes_dict.items()
      if v['category'] == 'issf'
  }
  issf_to_species = namespace.Mapping(
      'ebird2021_issf_to_ebird2021_species',
      'ebird2021_issf',
      'ebird2021_species',
      [(k, v) for (k, v) in sorted(issf_dict.items())],
  )

  def get_ebird_species_mapping(key, target_namespace):
    mapping_dict = {
        sp: codes_dict[sp][key]
        for sp in species
        if codes_dict[sp][key] in target_namespace.classes
    }
    mapping = namespace.Mapping.from_dict(
        f'ebird2021_species_to_{key}',
        'ebird2021_species',
        target_namespace.name,
        mapping_dict,
    )
    return mapping

  species_to_genus = get_ebird_species_mapping('genus', ebird_genera)
  species_to_family = get_ebird_species_mapping('family', ebird_families)
  species_to_order = get_ebird_species_mapping('order', ebird_orders)

  return GeneratorOutput(
      [
          clements_namespace,
          ebird_species,
          ebird_genera,
          ebird_families,
          ebird_orders,
          ebird_issf,
          ebird_all,
      ],
      [
          species_to_genus,
          species_to_family,
          species_to_order,
          issf_to_species,
          ebird_all_to_species,
          ebird_all_to_genus,
          ebird_all_to_family,
          ebird_all_to_order,
      ],
      [],
  )


def load_audioset_dict():
  """Load Audioset data in a convenient dictionary form."""
  audioset_fp = path_utils.get_absolute_epath(AUDIOSET_DATA_PATH)

  audioset = {}
  with open(audioset_fp, 'r') as f:
    data = json.load(f)

    for i in data:
      name = i['name'].replace(',', '_and')
      label = (name.strip()).replace(' ', '_')
      audioset[i['id']] = label

  return audioset


def generate_audioset():
  """Generate the Audioset namespace file."""
  # load the datasetfile
  audioset_dict = load_audioset_dict()

  audio_namespace = namespace.Namespace('audioset', set(audioset_dict.keys()))

  return GeneratorOutput([audio_namespace], [], [])
