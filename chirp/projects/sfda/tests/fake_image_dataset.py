# coding=utf-8
# Copyright 2023 The Perch Authors.
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

"""A fake image dataset for testing."""

import numpy as np
import tensorflow_datasets as tfds


class FakeImageDataset(tfds.core.GeneratorBasedBuilder):
  """Fake image dataset used for testing purposes."""

  VERSION = tfds.core.Version('1.0.0')

  def _split_generators(self, dl_manager):
    return {
        'train': self._generate_examples(10),
        'test': self._generate_examples(10),
    }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=['cat', 'dog']),
        }),
    )

  def _generate_examples(self, num_examples):
    for i in range(num_examples):
      yield i, {
          'image': np.zeros((12, 12, 3)).astype(np.uint8),
          'label': np.random.choice(self._info().features['label'].names),
      }
