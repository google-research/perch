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

"""Scrapes the Xeno-Canto website for taxonomy and audio data."""


import concurrent.futures
import enum
import functools
import itertools
import json
import os.path
from typing import Any, Sequence

from absl import app
from absl import flags
from chirp.data import utils
import pandas as pd
import ratelimiter
import requests
import tensorflow as tf
import tqdm

_XC_API_RATE_LIMIT = 8
_XC_API_URL = 'http://www.xeno-canto.org/api/2/recordings'
_XC_SPECIES_URL = 'https://xeno-canto.org/collection/species/all'


class _Modes(enum.Enum):
  COLLECT_INFO = 'collect_info'


_MODE = flags.DEFINE_enum(
    'mode', 'collect_info', [mode.value for mode in _Modes], 'Operation mode.'
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/xeno-canto',
    'Where to output the taxonomy info DataFrame.',
)
_INFO_FILENAME = flags.DEFINE_string(
    'info_filename', 'xeno_canto.jsonl', 'Xeno-Canto info filename.'
)


def collect_info(
    output_dir: str, recordings_filename: str
) -> list[dict[str, Any]]:
  """Scrapes the Xeno-Canto website for audio file IDs.

  Args:
    output_dir: Directory in which to store the list of recordings.
    recordings_filename: Filename to which to store recordings.

  Returns:
    The list of recordings.
  """
  # Collect all species
  (species,) = pd.read_html(io=_XC_SPECIES_URL, match='Scientific name')

  # Query Xeno-Canto for all recordings for each species
  session = requests.Session()
  session.mount(
      'http://',
      requests.adapters.HTTPAdapter(
          max_retries=requests.adapters.Retry(total=5, backoff_factor=0.1)
      ),
  )

  @ratelimiter.RateLimiter(max_calls=_XC_API_RATE_LIMIT, period=1)
  def get_recordings(scientific_name: str, page: int = 1):
    response = session.get(
        url=_XC_API_URL,
        params={
            'query': f"{scientific_name} gen:{scientific_name.split(' ')[0]}",
            'page': page,
        },
    )
    response.raise_for_status()
    results = response.json()['recordings']

    # Get next page if there are more
    if response.json()['numPages'] > page:
      results.extend(get_recordings(scientific_name, page + 1))
    return results

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=_XC_API_RATE_LIMIT
  ) as executor:
    species_recordings = executor.map(
        get_recordings, species['Scientific name']
    )
    species_recordings = tqdm.tqdm(
        species_recordings, total=len(species), desc='Collecting recordings'
    )
    recordings = list(itertools.chain.from_iterable(species_recordings))

  # Store recordings as JSONL file
  with tf.io.gfile.GFile(
      os.path.join(output_dir, recordings_filename), 'w'
  ) as f:
    for recording in recordings:
      f.write(json.dumps(recording))
      f.write('\n')
  return recordings


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  modes = {
      'collect_info': collect_info,
  }
  modes[_MODE.value](_OUTPUT_DIR.value, _INFO_FILENAME.value)


if __name__ == '__main__':
  app.run(main)
