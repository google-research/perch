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

"""Tests for namespace_db."""

import io
import random
import tempfile

from absl import logging
from hoplite.taxonomy import namespace
from hoplite.taxonomy import namespace_db
import numpy as np
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized


class NamespaceDbTest(parameterized.TestCase):

  def test_load_namespace_db(self):
    db = namespace_db.load_db()

    # Check a couple ClassLists of known size.
    self.assertIn('caples', db.class_lists)
    caples_list = db.class_lists['caples']
    self.assertEqual(caples_list.namespace, 'ebird2021')
    self.assertLen(caples_list.classes, 79)

    genus_mapping = db.mappings['ebird2021_to_genus']
    caples_genera = caples_list.apply_namespace_mapping(genus_mapping)
    self.assertEqual(caples_genera.namespace, 'ebird2021_genera')
    self.assertLen(caples_genera.classes, 62)

    family_mapping = db.mappings['ebird2021_to_family']
    caples_families = caples_list.apply_namespace_mapping(family_mapping)
    self.assertEqual(caples_families.namespace, 'ebird2021_families')
    self.assertLen(caples_families.classes, 30)

    order_mapping = db.mappings['ebird2021_to_order']
    caples_orders = caples_list.apply_namespace_mapping(order_mapping)
    self.assertEqual(caples_orders.namespace, 'ebird2021_orders')
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

  def test_class_map_csv(self):
    cl = namespace.ClassList(
        'ebird2021', ('amecro', 'amegfi', 'amered', 'amerob')
    )
    cl_csv = cl.to_csv()
    with io.StringIO(cl_csv) as f:
      got_cl = namespace.ClassList.from_csv(f)
    self.assertEqual(got_cl.namespace, 'ebird2021')
    self.assertEqual(got_cl.classes, ('amecro', 'amegfi', 'amered', 'amerob'))

    # Check that writing with tf.io.gfile behaves as expected, as newline
    # behavior may be different than working with StringIO.
    with tempfile.NamedTemporaryFile(suffix='.csv') as f:
      with tf.io.gfile.GFile(f.name, 'w') as gf:
        gf.write(cl_csv)
      with open(f.name, 'r') as f:
        got_cl = namespace.ClassList.from_csv(f.readlines())
    self.assertEqual(got_cl.namespace, 'ebird2021')
    self.assertEqual(got_cl.classes, ('amecro', 'amegfi', 'amered', 'amerob'))

  def test_namespace_class_list_closure(self):
    # Ensure that all classes in class lists appear in their namespace.
    db = namespace_db.load_db()

    all_missing_classes = set()
    for list_name, class_list in db.class_lists.items():
      missing_classes = set()
      namespace_ = db.namespaces[class_list.namespace]
      for cl in class_list.classes:
        if cl not in namespace_.classes:
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
    ebird = db.namespaces['ebird2021_species']
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

  def test_reef_label_converting(self):
    """Test operations used in ConvertReefLabels class.

    Part 1: Get the index of a sample in source_classes and the corresponding
     index from lookup table so that we can check the look up table returns
     the right soundtype e.g 'bioph' for the label 'bioph_rattle_response'.
    Part 2: Iterate over labels in a shuffled version of source_classes
     and check if each label maps correctly to its expected sound type.
    """
    # Set up
    db = namespace_db.load_db()
    mapping = db.mappings['reef_class_to_soundtype']
    source_classes = db.class_lists['all_reefs']
    target_classes = db.class_lists['all_reefs']
    soundtype_table = source_classes.get_namespace_map_tf_lookup(
        mapping, target_class_list=target_classes, keep_unknown=True
    )
    # Part 1
    test_labels = ['geoph_waves', 'bioph_rattle_response', 'anthrop_bomb']
    expected_results = ['geoph', 'bioph', 'anthrop']
    for test_label, expected_result in zip(test_labels, expected_results):
      classlist_index = source_classes.classes.index(test_label)
      lookup_index = soundtype_table.lookup(
          tf.constant(classlist_index, dtype=tf.int64)
      ).numpy()
      lookup_label = target_classes.classes[lookup_index]
      self.assertEqual(expected_result, lookup_label)
    # Part 2
    shuffled_classes = list(source_classes.classes)
    np.random.seed(42)
    random.shuffle(shuffled_classes)
    for label in shuffled_classes:
      # Every reef label is prefixed with either 'bioph', 'geoph', 'anthrop'
      prefix = label.split('_')[0]
      # Now mirror Part 1, by checking label against the prefix
      classlist_index = source_classes.classes.index(label)
      lookup_index = soundtype_table.lookup(
          tf.constant(classlist_index, dtype=tf.int64)
      ).numpy()
      lookup_label = target_classes.classes[lookup_index]
      self.assertEqual(prefix, lookup_label)

  @parameterized.parameters(True, False, None)
  def test_namespace_map_tf_lookup(self, keep_unknown):
    source = namespace.ClassList(
        'ebird2021', ('amecro', 'amegfi', 'amered', 'amerob', 'unknown')
    )
    mapping = namespace.Mapping(
        'ebird2021',
        'ebird2021',
        {
            'amecro': 'amered',
            'amegfi': 'amerob',
            'amered': 'amerob',
            'amerob': 'amerob',
        },
    )
    if keep_unknown is None:
      self.assertRaises(
          ValueError,
          source.get_namespace_map_tf_lookup,
          mapping=mapping,
          keep_unknown=keep_unknown,
      )
      return

    output_class_list = source.apply_namespace_mapping(
        mapping, keep_unknown=keep_unknown
    )
    if keep_unknown:
      expect_classes = ('amered', 'amerob', 'unknown')
    else:
      expect_classes = ('amered', 'amerob')
    self.assertSequenceEqual(output_class_list.classes, expect_classes)
    lookup = source.get_namespace_map_tf_lookup(
        mapping, keep_unknown=keep_unknown
    )
    got = lookup.lookup(
        tf.constant(list(range(len(source.classes))), dtype=tf.int64)
    ).numpy()
    if keep_unknown:
      expect_idxs = (0, 1, 1, 1, 2)
    else:
      expect_idxs = (0, 1, 1, 1, -1)
    self.assertSequenceEqual(tuple(got), expect_idxs)


if __name__ == '__main__':
  absltest.main()
