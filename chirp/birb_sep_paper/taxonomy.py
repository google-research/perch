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

"""Tools for managing taxonomy and enums.

Each tfRecord dataset encodes label, order, family, genus, and species with
separate enum values. This is a bit redundant, since we only really need to map
the label to the other parts of the taxonomy. However, it lets us avoid loading
up additional lookup tables when we ask the classifier to also predict other
levels of the taxonomy.

Basic idea for writing taxonomy enums:
* Construct a Taxonomy object T using a filepath containing only the
  SPECIES_INFO_CSV.
* Each species has a common name, species code, genus, family and order.
  The common name is hilariously unreliable, so we mainly use the species code
  as the primary identifier of a species. Every species maps uniquely to a
  genus, family, and order (in order of decreasing specificity).
* The SPECIES_INFO_CSV contains all info about all known species. Way more than
  you want.
* Then create your own filtered list of species codes.
  Feed them to T.GenerateEnums() to construct the enum lists, as described
  above, and write them out to disk if desired.

Then to use the enums, construct the Taxonomy object, by pointing it at a
directory containing all of the output enums.
"""

import csv
import os

from absl import logging
import dataset_info
import numpy as np

SPECIES = 'species'
LABEL = 'speciesCode'
GENUS = 'genus'
FAMILY = 'family'
ORDER = 'order'
COMMON_NAME = 'common_name'

UNKNOWN = 'unknown'
NONBIRD = 'nonbird'
HUMAN = 'human'


class Taxonomy(object):
  """Manages taxonomy info."""

  def __init__(self, model_path, data_path=None, species_info_path=None):
    self.model_path = model_path
    self.data_path = data_path
    self.species_info_path = species_info_path

    self.species_info = LoadSpeciesInfo(species_info_path)
    if os.path.exists(os.path.join(self.model_path, 'info.json')):
      self.LoadEnumsFromDatasetInfo()
    else:
      print('No info.json found.')

  def __getitem__(self, key):
    return self.species_info[key]

  def NumLabels(self):
    return len(self.label_enum) // 2

  def PrintEnumSizes(self):
    print('label  : ', len(self.label_enum) // 2)
    print('genus  : ', len(self.genus_enum) // 2)
    print('family : ', len(self.family_enum) // 2)
    print('order  : ', len(self.order_enum) // 2)

  def CommonNameToSpeciesCode(self, common_name):
    cn = common_name.strip().lower()
    for k, v in self.species_info.items():
      if v[COMMON_NAME] == cn:
        return k
      elif cn.replace('-', '') == k.replace('-', ''):
        return k
    return None

  def BackgroundSpeciesLookup(self, bg_string):
    """Get species label from a background species string."""
    if '(' not in bg_string:
      return self.CommonNameToSpeciesCode(bg_string)

    common, latin = bg_string.split('(')
    common = common.strip().lower()
    latin = latin.replace(')', '').strip().lower()
    genus = latin.split(' ')[0].lower()
    species = latin[len(genus) :].strip().lower()
    for k, v in self.species_info.items():
      if v[COMMON_NAME] == common:
        return k
    # Failed to find an exact match for the common name; try to find from latin.
    for k, v in self.species_info.items():
      if v[GENUS] != genus:
        continue
      if v[SPECIES].startswith(species) or species.startswith(v[SPECIES]):
        # Consider it a match.
        return k
    return None

  def BackgroundSpeciesToCodeList(self, bg_string, separator=';'):
    code_list = []
    for b in bg_string.split(separator):
      code = self.BackgroundSpeciesLookup(b)
      if code is not None:
        code_list.append(code)
    return code_list

  def CodeListToEnumList(self, code_list):
    """Convert list of codes to enum list."""
    enums = [self.label_enum.get(s, -1) for s in code_list]
    enums = [x for x in enums if x >= 0]
    return enums

  def LoadEnumsFromDatasetInfo(self):
    ds_info = dataset_info.read_dataset_info(self.model_path)
    self.label_enum = {i: k for (i, k) in enumerate(ds_info.label_set)}
    self.label_enum.update({k: i for (i, k) in enumerate(ds_info.label_set)})
    self.genus_enum = {i: k for (i, k) in enumerate(ds_info.genus_set)}
    self.genus_enum.update({k: i for (i, k) in enumerate(ds_info.genus_set)})
    self.family_enum = {i: k for (i, k) in enumerate(ds_info.family_set)}
    self.family_enum.update({k: i for (i, k) in enumerate(ds_info.family_set)})
    self.order_enum = {i: k for (i, k) in enumerate(ds_info.order_set)}
    self.order_enum.update({k: i for (i, k) in enumerate(ds_info.order_set)})

  def GenerateEnum(
      self, code_list, enum_type=ORDER, other_labels=None, code_whitelist=None
  ):
    """Create an Enum mapping for the provided list of species codes."""
    if other_labels is None:
      other_labels = [UNKNOWN, NONBIRD, HUMAN]
    code_list = sorted(code_list)
    enum = {}
    if code_whitelist:
      keys = [
          self.species_info[c][enum_type]
          for c in code_list
          if c in code_whitelist
      ]
    else:
      keys = [self.species_info[c][enum_type] for c in code_list]
    keys = sorted(list(set(keys)))
    keys = other_labels + keys
    for i, c in enumerate(keys):
      enum[c] = i
      enum[i] = c
    return enum

  def GenerateEnums(self, code_list, whitelist=None):
    """Generate enums from provided code list."""
    if whitelist is None:
      whitelist = {}
    self.label_enum = self.GenerateEnum(
        code_list, LABEL, code_whitelist=whitelist
    )
    self.order_enum = self.GenerateEnum(
        code_list, ORDER, code_whitelist=whitelist
    )
    self.family_enum = self.GenerateEnum(
        code_list, FAMILY, code_whitelist=whitelist
    )
    self.genus_enum = self.GenerateEnum(
        code_list, GENUS, code_whitelist=whitelist
    )

  def TranslateLabelVector(self, label_vector, other_taxonomy_path):
    """Convert a label vector from another taxonomy to this one."""
    taxo_old = Taxonomy(self.model_path, data_path=other_taxonomy_path)
    if label_vector.shape[1] != taxo_old.NumLabels():
      raise ValueError(
          'Label vector for conversion has shape %s, but '
          'the taxonomy has %d labels.'
          % (label_vector.shape, taxo_old.NumLabels())
      )
    trans = np.zeros(
        [taxo_old.NumLabels(), self.NumLabels()], label_vector.dtype
    )
    misses = []
    for i in range(taxo_old.NumLabels()):
      sp = taxo_old.label_enum[i]
      new_index = self.label_enum.get(sp, 0)
      if i > 0 and not new_index:
        misses.append(sp)
      trans[i, new_index] = 1
    labels_new = np.matmul(label_vector, trans)
    if misses:
      print('Some species were not in this taxonomy : %s' % misses)
    return labels_new

  def MakeSpeciesHints(
      self,
      species_list=None,
      species_list_tag='',
      dataset_info_path='',
      csv_path='',
  ):
    """Create a species hint vector from a provided list or taxonomy info."""
    if (
        species_list is None
        and not dataset_info_path
        and not csv_path
        and not species_list_tag
    ):
      logging.info('Using all-ones species hints.')
      return np.ones([self.NumLabels()], np.float32)

    hints = np.zeros([self.NumLabels()], np.float32)
    if species_list_tag:
      csv_fn = '%s_birds.csv' % species_list_tag.replace('taxo_', '')
      csv_fp = os.path.join(self.data_path, csv_fn)
      if os.path.exists(csv_fp):
        with open(csv_fp) as f:
          for r in f:
            sp = r.strip()
            if sp in self.label_enum:
              hints[self.label_enum[sp]] = 1
      else:
        raise ValueError(
            'File with the desired hints cannot be found : %s' % csv_fp
        )

    if csv_path:
      with open(csv_path) as f:
        for r in f:
          sp = r.strip()
          if sp in self.label_enum:
            hints[self.label_enum[sp]] = 1

    if dataset_info_path:
      ds_info = dataset_info.read_dataset_info(dataset_info_path)
      for sp in ds_info.label_set:
        if sp in self.label_enum:
          hints[self.label_enum[sp]] = 1

    if species_list:
      for sp in species_list:
        if sp in self.label_enum:
          hints[self.label_enum[sp]] = 1

    if np.sum(hints) == 0:
      raise ValueError('Tried loading hints, but no matches found.')
    logging.info('Loaded %d hints.', np.sum(hints))
    return hints


def LoadEnum(filepath):
  """Load an enum file into a two-way dict."""
  enum = {}
  with open(filepath) as c:
    for row in c:
      (i, v) = row.split(',')
      index = int(i)
      label = v.lower().strip()
      enum[index] = label
      enum[label] = index
  return enum


def LoadSpeciesInfo(species_info_path=None):
  """Load a dict mapping species codes to full taxonomy info for the species."""
  species_info = {}
  if species_info_path and os.path.exists(species_info_path):
    with open(species_info_path, 'rt') as c:
      species_info_raw = c.read()
      reader = csv.DictReader(species_info_raw.splitlines())
      for row in reader:
        species_info[row[LABEL]] = row
      for k in ['none', 'unknown', 'human']:
        species_info[k] = {
            COMMON_NAME: k,
            'enum': '0',
            FAMILY: k,
            GENUS: k,
            ORDER: k,
            SPECIES: k,
        }
  return species_info
