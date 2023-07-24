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

"""Tests for namespace_db."""

from absl import logging
from chirp.taxonomy import namespace_db
import numpy as np
import tensorflow as tf

from absl.testing import absltest


class NamespaceDbTest(absltest.TestCase):

  def test_load_namespace_db(self):
    db = namespace_db.load_db()
    for namespace_name in [
        'ebird2021',
        'bird_genera',
        'bird_families',
        'bird_orders',
    ]:
      self.assertIn(namespace_name, db.namespaces)
      # Also check that each namespace is represented as a ClassList.
      self.assertIn(namespace_name, db.class_lists)

    # Check a couple ClassLists of known size.
    self.assertIn('caples', db.class_lists)
    caples_list = db.class_lists['caples']
    self.assertEqual(caples_list.namespace, 'ebird2021')
    self.assertLen(caples_list.classes, 79)

    genus_mapping = db.mappings['ebird2021_to_genus']
    caples_genera = caples_list.apply_namespace_mapping(genus_mapping)
    self.assertEqual(caples_genera.namespace, 'bird_genera')
    self.assertLen(caples_genera.classes, 62)

    family_mapping = db.mappings['ebird2021_to_family']
    caples_families = caples_list.apply_namespace_mapping(family_mapping)
    self.assertEqual(caples_families.namespace, 'bird_families')
    self.assertLen(caples_families.classes, 30)

    order_mapping = db.mappings['ebird2021_to_order']
    caples_orders = caples_list.apply_namespace_mapping(order_mapping)
    self.assertEqual(caples_orders.namespace, 'bird_orders')
    self.assertLen(caples_orders.classes, 11)

  def test_class_maps(self):
    db = namespace_db.load_db()
    caples_list = db.class_lists['caples']
    sierras_list = db.class_lists['sierra_nevadas']
    table, image_mask = caples_list.get_class_map_tf_lookup(sierras_list)
    # The Caples list is a strict subset of the Sierras list.
    self.assertLen(caples_list.classes, np.sum(image_mask))
    self.assertEqual(image_mask.shape, (len(sierras_list.classes),))
    for i in range(len(caples_list.classes)):
      self.assertGreaterEqual(
          table.lookup(tf.constant([i], dtype=tf.int64)).numpy()[0], 0
      )

  def test_namespace_class_list_closure(self):
    # Ensure that all classes in class lists appear in their namespace.
    db = namespace_db.load_db()

    all_missing_classes = set()
    for list_name, class_list in db.class_lists.items():
      missing_classes = set()
      namespace = db.namespaces[class_list.namespace]
      for cl in class_list.classes:
        if cl not in namespace.classes:
          missing_classes.add(cl)
          all_missing_classes.add(cl)
      if missing_classes:
        logging.warning(
            'The classes %s in class list %s did not appear in namespace %s.',
            missing_classes,
            list_name,
            class_list.namespace,
        )
      missing_classes.discard('unknown')
    all_missing_classes.discard('unknown')
    self.assertEmpty(all_missing_classes)

  def test_namespace_mapping_closure(self):
    # Ensure that all classes in mappings appear in their namespace.
    db = namespace_db.load_db()

    all_missing_classes = set()
    for mapping_name, mapping in db.mappings.items():
      missing_source_classes = set()
      missing_target_classes = set()
      source_namespace = db.namespaces[mapping.source_namespace]
      target_namespace = db.namespaces[mapping.target_namespace]
      for source_cl, target_cl in mapping.mapped_pairs.items():
        if source_cl not in source_namespace.classes:
          missing_source_classes.add(source_cl)
          all_missing_classes.add(source_cl)
        if target_cl not in target_namespace.classes:
          missing_target_classes.add(target_cl)
          all_missing_classes.add(target_cl)
      if missing_source_classes:
        logging.warning(
            'The classes %s in mapping %s did not appear in namespace %s.',
            missing_source_classes,
            mapping_name,
            source_namespace.name,
        )
      if missing_target_classes:
        logging.warning(
            'The classes %s in mapping %s did not appear in namespace %s.',
            missing_target_classes,
            mapping_name,
            target_namespace.name,
        )
      missing_target_classes.discard('unknown')
    self.assertEmpty(all_missing_classes)

  def test_taxonomic_mappings(self):
    # Ensure that all ebird2021 species appear in taxonomic mappings.
    db = namespace_db.load_db()
    ebird = db.class_lists['ebird2021_species']
    genera = db.mappings['ebird2021_to_genus'].mapped_pairs
    families = db.mappings['ebird2021_to_family'].mapped_pairs
    orders = db.mappings['ebird2021_to_order'].mapped_pairs
    missing_genera = set()
    missing_families = set()
    missing_orders = set()
    for cl in ebird.classes:
      if cl not in genera:
        missing_genera.add(cl)
      if cl not in families:
        missing_families.add(cl)
      if cl not in orders:
        missing_orders.add(cl)
    self.assertEmpty(missing_genera)
    self.assertEmpty(missing_families)
    self.assertEmpty(missing_orders)


if __name__ == '__main__':
  absltest.main()
