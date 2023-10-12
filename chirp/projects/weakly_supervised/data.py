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

"""Data."""
from chirp.data import google_grain_utils
from chirp.taxonomy import namespace_db
from grain import python as pygrain
import jax
from ml_collections import config_dict


def get_dataset(config: config_dict.ConfigDict) -> pygrain.DataLoader:
  """Load the dataset using Grain."""
  # Load the dataset
  data_source = pygrain.ArrayRecordDataSource(config.dataset)
  num_records = len(data_source)

  shard_options = pygrain.ShardOptions(
      shard_index=jax.process_index(),
      shard_count=jax.process_count(),
      drop_remainder=True,
  )
  sampler = pygrain.IndexSampler(
      num_records=num_records,
      shard_options=shard_options,
      seed=0,
      shuffle=True,
  )

  # Pipeline
  db = namespace_db.load_db()
  class_list = db.class_lists[config.source_class_list]
  operations_ = [
      google_grain_utils.Parse(),
      google_grain_utils.MultiHot(class_list.namespace, class_list.classes),
      google_grain_utils.Window(config.window_size, config.hop_size),
      google_grain_utils.BinPacking(
          batch_size=config.batch_size,
          num_bins=jax.local_device_count(),
          size_fn=lambda x: len(x["windows"]),
      ),
  ]
  return pygrain.DataLoader(
      data_source=data_source,
      sampler=sampler,
      operations=operations_,
      worker_count=config.worker_count,
      shard_options=shard_options,
  )
