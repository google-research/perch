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

"""A fake dataset for testing."""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class FakeDataset(tfds.core.GeneratorBasedBuilder):
  """Fake dataset."""

  VERSION = tfds.core.Version('1.0.0')

  LABEL_NAMES = [str(i) for i in range(90)]
  GENUS_NAMES = [str(i) for i in range(60)]
  FAMILY_NAMES = [str(i) for i in range(30)]
  ORDER_NAMES = [str(i) for i in range(20)]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'audio':
                tfds.features.Audio(dtype=tf.float32, sample_rate=44_100),
            'label':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=self.LABEL_NAMES)),
            'label_str':
                tfds.features.Sequence(tfds.features.Text()),
            'bg_labels':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=self.LABEL_NAMES)),
            'genus':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=self.GENUS_NAMES)),
            'family':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=self.FAMILY_NAMES)),
            'order':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=self.ORDER_NAMES)),
            'filename':
                tfds.features.Text(),
        }),
        description='Fake dataset.',
    )

  def _split_generators(self, dl_manager):
    return {
        'train': self._generate_examples(100),
        'test': self._generate_examples(20),
    }

  def _generate_examples(self, num_examples):
    for i in range(num_examples):
      yield i, {
          'audio': np.random.uniform(-1.0, 1.0, [44_100]),
          'label': np.random.choice(self.LABEL_NAMES, size=[3]),
          'label_str': ['placeholder'] * 3,
          'bg_labels': np.random.choice(self.LABEL_NAMES, size=[2]),
          'genus': np.random.choice(self.GENUS_NAMES, size=[1]),
          'family': np.random.choice(self.FAMILY_NAMES, size=[1]),
          'order': np.random.choice(self.ORDER_NAMES, size=[2]),
          'filename': 'placeholder',
      }
