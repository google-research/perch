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

"""Utilities for data processing."""
import hashlib
import os.path


def xeno_canto_filename(filename: str, id_: int) -> tuple[str, str]:
  """Determine a filename for a Xeno-Canto recording.

  We can't use the original filename since some of those are not valid Unix
  filenames (e.g., they contain slashes). Hence the files are named using just
  their ID. There are some files with spaces in the extension, so that is
  handled here as well.

  We also return the first two characters of the MD5 hash of the filename. This
  can be used to evenly distribute files across 256 directories in a
  deterministic manner.

  Args:
    filename: The original filename (used to determine the extension).
    id_: The Xeno-Canto ID of the recording.

  Returns:
    A tuple where the first element is the filename to save this recording two
    and the second element is a two-character subdirectory name in which to
    save the file.
  """
  # Two files have the extension ". mp3"
  ext = os.path.splitext(filename)[1].lower().replace(' ', '')
  filename = f'XC{id_}{ext}'
  # Results in ~2900 files per directory
  subdir = hashlib.md5(filename.encode()).hexdigest()[:2]
  return filename, subdir
