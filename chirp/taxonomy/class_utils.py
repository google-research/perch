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

"""Convenience utilities for handling class lists."""

from chirp.taxonomy import namespace_db


def get_class_lists(species_class_list_name: str, add_taxonomic_labels: bool):
  """Get the number of classes for the target class outputs."""
  db = namespace_db.load_db()
  species_classes = db.class_lists[species_class_list_name]
  class_lists = {
      "label": species_classes,
  }
  if add_taxonomic_labels:
    for name in ["genus", "family", "order"]:
      mapping_name = f"{species_classes.namespace}_to_{name}"
      mapping = db.mappings[mapping_name]
      taxa_class_list = species_classes.apply_namespace_mapping(mapping)
      class_lists[name] = taxa_class_list
  return class_lists
