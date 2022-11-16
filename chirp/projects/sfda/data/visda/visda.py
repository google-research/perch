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

"""VisDa-C dataset."""

import itertools

import tensorflow_datasets as tfds

_DESCRIPTION = """
VisDa-C dataset.
"""

_CITATION = """
@article{peng2017visda,
  title={Visda: The visual domain adaptation challenge},
  author={Peng, Xingchao and Usman, Ben and Kaushik, Neela and Hoffman, Judy and
          Wang, Dequan and Saenko, Kate},
  journal={arXiv preprint arXiv:1710.06924},
  year={2017}
}
"""


class VisDaC(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for the VisDa-C dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  TRAIN_DATA_URL = 'http://csr.bu.edu/ftp/visda17/clf/train.tar'
  VALIDATION_DATA_URL = 'http://csr.bu.edu/ftp/visda17/clf/validation.tar'

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image':
                tfds.features.Image(),
            'label':
                tfds.features.ClassLabel(names=[
                    'aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
                    'motorcycle', 'person', 'plant', 'skateboard', 'train',
                    'truck'
                ]),
        }),
        supervised_keys=('image', 'label'),
        homepage=('https://github.com/VisionLearningGroup/taskcv-2017-public/'
                  'tree/master/classification'),
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    paths = dl_manager.download_and_extract({
        'train': self.TRAIN_DATA_URL,
        'validation': self.VALIDATION_DATA_URL,
    })
    return {
        'train':
            self._generate_examples(data_path=paths['train'] / 'train'),
        'validation':
            self._generate_examples(data_path=paths['validation'] / 'validation'
                                   ),
    }

  def _generate_examples(self, data_path):
    counter = itertools.count()

    for path in data_path.iterdir():
      if path.is_dir():
        for image_path in path.iterdir():
          yield next(counter), {
              'image': image_path,
              'label': path.name,
          }
