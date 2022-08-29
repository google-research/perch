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

"""Tests for namespace_db."""

from chirp.taxonomy import namespace_db
import numpy as np
import tensorflow as tf

from absl.testing import absltest


class NamespaceDbTest(absltest.TestCase):

  def test_load_namespace_db(self):
    db = namespace_db.NamespaceDatabase.load_csvs()
    for namespace_name in [
        'ebird2021', 'bird_genera', 'bird_families', 'bird_orders'
    ]:
      self.assertIn(namespace_name, db.namespaces)
      # Also check that each namespace is represented as a ClassList.
      self.assertIn(namespace_name, db.class_lists)
      self.assertEqual(namespace_name, db.class_lists[namespace_name].name)

    # Check a couple ClassLists of known size.
    self.assertIn('caples', db.class_lists)
    caples_list = db.class_lists['caples']
    self.assertEqual(caples_list.namespace, 'ebird2021')
    self.assertEqual(caples_list.size, 79)

    genus_mapping = db.mappings['ebird2021_to_genus']
    caples_genera = caples_list.apply_namespace_mapping(genus_mapping,
                                                        'caples_genera')
    self.assertEqual(caples_genera.name, 'caples_genera')
    self.assertEqual(caples_genera.namespace, 'bird_genera')
    self.assertEqual(caples_genera.size, 62)

    family_mapping = db.mappings['ebird2021_to_family']
    caples_families = caples_list.apply_namespace_mapping(
        family_mapping, 'caples_families')
    self.assertEqual(caples_families.name, 'caples_families')
    self.assertEqual(caples_families.namespace, 'bird_families')
    self.assertEqual(caples_families.size, 30)

    order_mapping = db.mappings['ebird2021_to_order']
    caples_orders = caples_list.apply_namespace_mapping(order_mapping,
                                                        'caples_orders')
    self.assertEqual(caples_orders.name, 'caples_orders')
    self.assertEqual(caples_orders.namespace, 'bird_orders')
    self.assertEqual(caples_orders.size, 11)

  def test_class_maps(self):
    db = namespace_db.NamespaceDatabase.load_csvs()
    caples_list = db.class_lists['caples']
    sierras_list = db.class_lists['sierra_nevadas']
    table, image_mask = caples_list.get_class_map_tf_lookup(sierras_list)
    # The Caples list is a strict subset of the Sierras list.
    self.assertEqual(np.sum(image_mask), caples_list.size)
    self.assertEqual(image_mask.shape, (sierras_list.size,))
    for i in range(caples_list.size):
      self.assertGreaterEqual(
          table.lookup(tf.constant([i], dtype=tf.int64)).numpy()[0], 0)


if __name__ == '__main__':
  absltest.main()
