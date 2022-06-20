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

from chirp.data import bird_taxonomy
import numpy as np


class FakeDataset(bird_taxonomy.BirdTaxonomy):
  """Fake dataset."""

  def _split_generators(self, dl_manager):
    return {
        'train': self._generate_examples(100),
        'train_reduced': self._generate_examples(100, True),
        'test': self._generate_examples(20),
    }

  def _generate_examples(self, num_examples: int, reduced=False):
    for i in range(num_examples):
      record = {
          'audio':
              np.random.uniform(-1.0, 0.99,
                                self.info.features['audio'].shape).astype(
                                    np.float32),
          'filename':
              'placeholder',
          'quality_score':
              np.random.choice(['A', 'B', 'C', 'D', 'E', 'F']),
          'license':
              '//creativecommons.org/licenses/by-nc-sa/4.0/',
          'country':
              np.random.choice(['South Africa', 'Colombia', 'Namibia']),
          'altitude':
              str(np.random.uniform(0, 3000)),
          'bird_seen':
              np.random.choice(['yes', 'no']),
          'latitude':
              str(np.random.uniform(0, 90)),
          'longitude':
              str(np.random.uniform(0, 90)),
          'playback_used':
              'yes',
          'recordist':
              'N/A',
          'remarks':
              'N/A',
          'sound_type':
              np.random.choice(['call', 'song'])
      }
      if reduced:
        record.update({
            'label':
                np.random.choice(
                    self.info.features['label'].names[:5], size=[3]),
            'bg_labels':
                np.random.choice(
                    self.info.features['bg_labels'].names[:5], size=[2]),
            'genus':
                np.random.choice(
                    self.info.features['genus'].names[:5], size=[3]),
            'family':
                np.random.choice(
                    self.info.features['family'].names[:5], size=[3]),
            'order':
                np.random.choice(
                    self.info.features['order'].names[:5], size=[3]),
        })
      else:
        record.update({
            'label':
                np.random.choice(self.info.features['label'].names, size=[3]),
            'bg_labels':
                np.random.choice(
                    self.info.features['bg_labels'].names, size=[2]),
            'genus':
                np.random.choice(self.info.features['genus'].names, size=[3]),
            'family':
                np.random.choice(self.info.features['family'].names, size=[3]),
            'order':
                np.random.choice(self.info.features['order'].names, size=[3]),
        })
      yield i, record
