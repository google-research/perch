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

"""Utility functions for working with a Bioacoustic Workbench (baw) (e.g. A2O) API."""

import io
import os
import re
from typing import Generator, Sequence
import urllib

from chirp import audio_utils
import librosa
from ml_collections import config_dict
import numpy as np
import requests
import soundfile


def make_baw_audio_url_from_file_id(
    file_id: str, offset_s: float, window_size_s: float, baw_domain: str = "data.acousticsobervatory.org"
) -> str:
  """Construct an baw audio URL."""
  # Extract the recording UID. Example:
  # 'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.flac' -> 909057
  # 'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.wav' -> 909057
  pattern = re.compile(r'.*_(\d+)\.[^\.]+$')
  match = pattern.search(file_id)
  if not match:
      raise ValueError("Invalid file_id format")
  file_id = match.group(1)
  offset_s = int(offset_s)
  # See: https://api.staging.ecosounds.org/api-docs/index.html
  audio_path = (
      f"https://{baw_domain}/audio_recordings/"
      f"{file_id}/media.flac"
  )
  if offset_s <= 0 and window_size_s <= 0:
    return audio_path
  params = {}
  if offset_s > 0:
    params["start_offset"] = offset_s
  if window_size_s > 0:
    params["end_offset"] = offset_s + int(window_size_s)
  audio_path = audio_path + "?" + urllib.parse.urlencode(params)
  return audio_path


def load_baw_audio(
    audio_url: str,
    auth_token: str,
    sample_rate: int,
    session: requests.Session,
) -> np.ndarray | None:
  """Load audio from the baw API.

  Args:
    audio_url: URL to load the audio from.
    auth_token: The baw API auth token.
    sample_rate: The sample rate to resample the audio to.
    session: The requests session to use.

  Returns:
    The audio as a numpy array, or None if the audio could not be loaded.
  """

  if session is None:
    # Use requests.get instead of session.get if no session is provided.
    session = requests
  audio_response = session.get(
      url=audio_url,
      headers={"Authorization": f"Token token={auth_token}"},
  )
  if not audio_response.ok:
    print(audio_response.status_code)
    return None

  # Load the audio and resample.
  try:
    with io.BytesIO(audio_response.content) as f:
      sf = soundfile.SoundFile(f)
      audio = sf.read()
      audio = librosa.resample(
          audio, orig_sr=sf.samplerate, target_sr=sample_rate
      )
  except soundfile.LibsndfileError:
    return None
  return audio


def multi_load_baw_audio(
    filepaths: Sequence[str],
    offsets: Sequence[int],
    auth_token: str,
    sample_rate: int = 32000,
    **kwargs,
) -> Generator[np.ndarray, None, None]:
  """Creates a generator that loads audio from the baw API."""
  session = requests.Session()
  session.mount(
      "https://",
      requests.adapters.HTTPAdapter(
          max_retries=requests.adapters.Retry(total=5, backoff_factor=0.5)
      ),
  )
  baw_audio_loader = lambda fp, offset: load_baw_audio(
      fp, sample_rate=sample_rate, auth_token=auth_token, session=session
  )
  iterator = audio_utils.multi_load_audio_window(
      filepaths=filepaths,
      offsets=offsets,
      audio_loader=baw_audio_loader,
      **kwargs,
  )
  try:
    for ex in iterator:
      yield ex
  finally:
    session.close()
