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

"""Tests for eval_lib."""

import tempfile
from typing import Any, Sequence, Tuple

from chirp import config_utils
from chirp.configs import config_globals
from chirp.data.bird_taxonomy import bird_taxonomy
from chirp.eval import eval_lib
from chirp.tests import fake_dataset
import ml_collections
import tensorflow as tf

from absl.testing import absltest

_c = config_utils.callable_config


def _stub_localization_fn(audio: Any,
                          sample_rate_hz: int,
                          interval_length_s: float = 6.0,
                          max_intervals: int = 5) -> Sequence[Tuple[int, int]]:
  # The only purpose of this stub function is to avoid a default
  # `localization_fn` value of None in `BirdTaxonomyConfig` so that the audio
  # feature shape gets computed properly.
  del audio, sample_rate_hz, interval_length_s, max_intervals
  return []


class FakeBirdTaxonomy(fake_dataset.FakeDataset):
  BUILDER_CONFIGS = [
      bird_taxonomy.BirdTaxonomyConfig(
          name='fake_variant_1',
          localization_fn=_stub_localization_fn,
          interval_length_s=6.0),
      bird_taxonomy.BirdTaxonomyConfig(
          name='fake_variant_2',
          localization_fn=_stub_localization_fn,
          interval_length_s=6.0),
  ]


class LoadEvalDatasetsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.data_dir = tempfile.TemporaryDirectory('data_dir').name
    FakeBirdTaxonomy(
        data_dir=self.data_dir, config='fake_variant_1').download_and_prepare()
    FakeBirdTaxonomy(
        data_dir=self.data_dir, config='fake_variant_2').download_and_prepare()

  def test_return_value_structure(self):
    fake_config = ml_collections.ConfigDict()
    fake_config.dataset_configs = {
        'fake_dataset_1': {
            'tfds_name':
                'fake_bird_taxonomy/fake_variant_1',
            'tfds_data_dir':
                self.data_dir,
            'pipeline':
                _c('pipeline.Pipeline', ops=[_c('pipeline.OnlyJaxTypes')]),
            'split':
                'train',
        },
        'fake_dataset_2': {
            'tfds_name':
                'fake_bird_taxonomy/fake_variant_2',
            'tfds_data_dir':
                self.data_dir,
            'pipeline':
                _c('pipeline.Pipeline', ops=[_c('pipeline.OnlyJaxTypes')]),
            'split':
                'train',
        },
    }
    fake_config = config_utils.parse_config(fake_config,
                                            config_globals.get_globals())
    eval_datasets = eval_lib.load_eval_datasets(fake_config)

    self.assertSameElements(['fake_dataset_1', 'fake_dataset_2'],
                            eval_datasets.keys())
    for dataset in eval_datasets.values():
      self.assertIsInstance(dataset, tf.data.Dataset)
      self.assertContainsSubset(['audio', 'label', 'bg_labels'],
                                dataset.element_spec.keys())


if __name__ == '__main__':
  absltest.main()
