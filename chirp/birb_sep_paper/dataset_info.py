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

"""Descriptive dataset info."""

import dataclasses
import json
import os
import tensorflow as tf


@dataclasses.dataclass()
class DatasetInfo:
  """Describes dataset contents."""

  sample_rate_hz: int = 22050
  example_size_s: float = 6.0
  comment: str = ''
  label_set: tuple[str, ...] = ()
  genus_set: tuple[str, ...] = ()
  family_set: tuple[str, ...] = ()
  order_set: tuple[str, ...] = ()
  train_sstables: str = 'train_xc/tf.sstable-*'
  noise_sstables: str = 'noise/tf.sstable-*'
  eval_sstables: str = 'eval_xc/tf.sstable-*'
  eval_ss_sstables: str = 'eval_ss/tf.sstable-*'
  species_info_csv: str = 'species_info.csv'

  def add_enums_from_taxonomy(self, taxo):
    self.label_set = tuple(
        [taxo.label_enum[i] for i in range(len(taxo.label_enum) // 2)]
    )
    self.genus_set = tuple(
        [taxo.genus_enum[i] for i in range(len(taxo.genus_enum) // 2)]
    )
    self.family_set = tuple(
        [taxo.family_enum[i] for i in range(len(taxo.family_enum) // 2)]
    )
    self.order_set = tuple(
        [taxo.order_enum[i] for i in range(len(taxo.order_enum) // 2)]
    )

  def write(self, output_path, filename='info.json'):
    with tf.io.gfile.GFile(os.path.join(output_path, filename), 'w') as f:
      f.write(json.dumps(dataclasses.asdict(self)))


def read_dataset_info(info_path, filename='info.json'):
  with tf.io.gfile.GFile(os.path.join(info_path, filename), 'r') as f:
    data = json.loads(f.read())
  return DatasetInfo(**data)
