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

"""Ingest fully-annotated dataset labels from CSV."""

import dataclasses
from typing import Callable, Sequence

from chirp.projects.hoplite import interface
from chirp.taxonomy import annotations
from etils import epath
import pandas as pd
import tqdm


@dataclasses.dataclass
class AnnotatedDatasetIngestor:
  """Add annotations to embeddings DB from CSV annotations.

  Note that currently we only add positive labels.
  """

  base_path: str
  dataset_name: str
  annotation_filename: str
  annotation_load_fn: Callable[[str | epath.Path], pd.DataFrame]
  window_size_s: float
  provenance: str = 'dataset'

  def ingest_dataset(self, db: interface.GraphSearchDBInterface) -> set[str]:
    """Load annotations and insert labels into the DB."""
    annos_path = epath.Path(self.base_path) / self.annotation_filename
    annos_df = self.annotation_load_fn(annos_path)
    lbl_count = 0
    file_ids = annos_df['filename'].unique()
    label_set = set()
    for file_id in tqdm.tqdm(file_ids):
      embedding_ids = db.get_embeddings_by_source(self.dataset_name, file_id)
      source_annos = annos_df[annos_df['filename'] == file_id]
      for idx in embedding_ids:
        source = db.get_embedding_source(idx)
        window_start = source.offsets[0]
        window_end = window_start + self.window_size_s
        emb_annos = source_annos[source_annos['start_time_s'] < window_end]
        emb_annos = emb_annos[emb_annos['end_time_s'] > window_start]
        # All of the remianing annotations match the target embedding.
        for labels in emb_annos['label']:
          for label in labels:
            label_set.add(label)
            lbl = interface.Label(
                idx, label, interface.LabelType.POSITIVE, self.provenance
            )
            db.insert_label(lbl)
            lbl_count += 1
    print(f'\nInserted {lbl_count} labels.')
    return label_set
