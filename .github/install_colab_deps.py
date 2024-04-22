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

"""Installs Colab dependencies for CI testing."""

from typing import Sequence

from absl import app
import requests


REQS_FILE = 'https://raw.githubusercontent.com/googlecolab/backend-info/main/pip-freeze.txt'
COLAB_REQS_FILE = '/tmp/colab_reqs.txt'


def main(unused_argv: Sequence[str]) -> None:
  got = requests.get(REQS_FILE)
  requirements_str = str(got.content, 'utf8')
  # Skip the file:// lines, which we do not have access to.
  lines = [
      ln + '\n' for ln in requirements_str.split('\n') if 'file://' not in ln
  ]
  with open(COLAB_REQS_FILE, 'w') as f:
    f.writelines(lines)


if __name__ == '__main__':
  app.run(main)
