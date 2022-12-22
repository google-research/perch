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

"""Generates the global_seabirds class list.

Dynamically generates the list of ebird2021 labels that are considered
'seabirds' by extracting ebird2021 labels that belong to specific taxonomic
categories, which are specified in various seabird_<namespace> class lists.
"""
from typing import Sequence
from absl import app

from chirp import path_utils
from chirp.taxonomy import namespace
from chirp.taxonomy import namespace_db

OUTPUT = 'taxonomy/data/class_lists/global_seabirds.csv'

EBIRD_MAPPINGS = ['ebird2021_to_order', 'ebird2021_to_family']
SEABIRD_CLASS_LISTS = ['seabird_orders', 'seabird_families']


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  db = namespace_db.load_db()

  mappings = {
      db.mappings[db_key].target_namespace: db.mappings[db_key]
      for db_key in EBIRD_MAPPINGS
  }
  seabird_class_lists_by_namespace = {
      db.class_lists[class_list].namespace:
      set(db.class_lists[class_list].classes)
      for class_list in SEABIRD_CLASS_LISTS
  }

  ebirds = []
  for ns, mapping in mappings.items():
    seabird_classes = seabird_class_lists_by_namespace[ns]
    ebirds.extend(
        ebird for (ebird, classname) in mapping.mapped_pairs
        if classname in seabird_classes)

  seabirds = namespace.ClassList('_', 'ebird2021', ebirds)
  with open(path_utils.get_absolute_epath(OUTPUT), 'w') as f:
    f.write(seabirds.to_csv())


if __name__ == '__main__':
  app.run(main)
