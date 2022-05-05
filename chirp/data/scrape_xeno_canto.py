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

"""Scrapes the Xeno-Canto website for taxonomy and audio data."""

from typing import Sequence

from absl import app
from absl import flags
from chirp.data import xeno_canto
from etils import epath
import pandas as pd

_MODES = (
    'collect_info',
    'collect_info_and_download_audio',  # copypara:strip
)
_MODE = flags.DEFINE_enum('mode', 'collect_info', _MODES, 'Operation mode.')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', '/tmp/xeno-canto',
    'Where to output the taxonomy info DataFrame.')
_TAXONOMY_INFO_FILENAME = flags.DEFINE_string('taxonomy_info_filename',
                                              'taxonomy_info.json',
                                              'Taxonomy info filename.')


def collect_info(output_dir: str, taxonomy_info_filename: str) -> None:
  """Scrapes the Xeno-Canto website for audio file IDs.

  Args:
    output_dir: output directory for the taxonomy info DataFrame.
    taxonomy_info_filename: taxonomy info filename.
  """
  taxonomy_info = xeno_canto.create_taxonomy_info(
      xeno_canto.SpeciesMappingConfig())
  taxonomy_info = xeno_canto.retrieve_recording_metadata(taxonomy_info)
  with (epath.Path(output_dir) / taxonomy_info_filename).open('w') as f:
    taxonomy_info.to_json(f)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  modes = {
      'collect_info': collect_info,
  }
  modes[_MODE.value](_OUTPUT_DIR.value, _TAXONOMY_INFO_FILENAME.value)


if __name__ == '__main__':
  app.run(main)
