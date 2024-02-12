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

"""Scrape iNaturalist for audio recordings."""

from collections.abc import Sequence
import csv
import math
import os.path
import time
import urllib.parse

from absl import app
from absl import flags
from absl import logging
import ratelimiter
import requests
import tensorflow as tf
import tqdm

_INPUT_FILE = flags.DEFINE_string(
    "input_file",
    "inaturalist.csv",
    "A CSV file containing an identifier column of files to download.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "/tmp/inaturalist",
    "Where to store the metadata and audio.",
)


def download_audio(input_file, output_dir):
  """Download all audio observations in iNaturalist.

  The regular API does not allow loading more than 10,000 observations and the
  export tool doesn't allow more than 200,000. Hence we use the export from
  iNaturalist that is submitted to GBIF. Note that this is a more limited set of
  recordings consisting of CC-BY, CC-BY-NC, and CC0 recordings which are
  "research-grade".

  Note that there are ~400,000 research-grade recordings of birds, but only
  ~300,000 of those have a CC-BY, CC-BY-NC or CC0 license according to the API.
  The majority (~80,000) of the others simply have no license attached.

  We use the iNaturalist export directly rather than going through GBIF because
  that export file is easier to download (even though it's bigger and needs to
  be filtered). This function assumes that `inaturalist.sh` has been run on the
  this file: http://www.inaturalist.org/observations/gbif-observations-dwca.zip.

  Args:
    input_file: A CSV file which contains a column called "identifier" with the
      URLs of files to download.
    output_dir: A writable directory in which the audio files will be stored.
  """
  # Set download limits
  day_limit = ratelimiter.RateLimiter(
      max_calls=24 * 1024**2, period=24 * 60 * 60  # 24 GB per day
  )
  hour_limit = ratelimiter.RateLimiter(
      max_calls=5 * 1024**2, period=60 * 60  # 5 GB per hour
  )

  # Retry downloads
  session = requests.Session()
  session.mount(
      "https://",
      requests.adapters.HTTPAdapter(
          max_retries=requests.adapters.Retry(total=5, backoff_factor=0.1)
      ),
  )

  def download_file(row):
    filename = os.path.basename(urllib.parse.urlparse(row["identifier"]).path)
    audio_target = os.path.join(output_dir, filename)
    if not tf.io.gfile.exists(audio_target):
      with tf.io.gfile.GFile(audio_target, "wb") as f:
        r = session.get(url=row["identifier"])
        if not r.ok:
          return False
        f.write(r.content)
        # Increment the rate limiter once for each kilobyte downloaded
        for _ in range(math.ceil(len(r.content) / 1024)):
          with day_limit, hour_limit:
            pass
    return True

  for row in tqdm.tqdm(
      csv.DictReader(tf.io.gfile.GFile(input_file)),
      total=sum(1 for _ in tf.io.gfile.GFile(input_file)),
  ):
    for num_try in range(math.ceil(math.log2(60 * 60 * 24))):
      if download_file(row):
        break
      else:
        logging.warning(
            "Failed to download %s, sleeping %s seconds...",
            row["id"],
            2**num_try,
        )
        time.sleep(2**num_try)
    else:
      logging.error("Failed to download %s, moving on", row["id"])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  download_audio(_INPUT_FILE.value, _OUTPUT_DIR.value)


if __name__ == "__main__":
  app.run(main)
