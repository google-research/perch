# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""Office-31 dataset."""

import itertools

import tensorflow_datasets as tfds

_DESCRIPTION = """
Office-31 dataset.
"""

_CITATION = """
@inproceedings{saenko2010adapting,
  title={Adapting visual category models to new domains},
  author={Saenko, Kate and Kulis, Brian and Fritz, Mario and Darrell, Trevor},
  booktitle={Proceedings of the European Conference on Computer Vision},
  pages={213--226},
  year={2010},
}
"""


class Office31(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for the Office-31 dataset."""

  name = 'office_31'
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  DATA_URL = ('https://drive.google.com/uc?export=download&'
              'id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE')
  BUILDER_CONFIGS = [
      tfds.core.BuilderConfig(name=name)
      for name in ('amazon', 'dslr', 'webcam')
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'label': tfds.features.ClassLabel(
                names=[
                    'back_pack',
                    'bike',
                    'bike_helmet',
                    'bookcase',
                    'bottle',
                    'calculator',
                    'desk_chair',
                    'desk_lamp',
                    'desktop_computer',
                    'file_cabinet',
                    'headphones',
                    'keyboard',
                    'laptop_computer',
                    'letter_tray',
                    'mobile_phone',
                    'monitor',
                    'mouse',
                    'mug',
                    'paper_notebook',
                    'pen',
                    'phone',
                    'printer',
                    'projector',
                    'punchers',
                    'ring_binder',
                    'ruler',
                    'scissors',
                    'speaker',
                    'stapler',
                    'tape_dispenser',
                    'trash_can',
                ]
            ),
        }),
        supervised_keys=('image', 'label'),
        homepage='https://faculty.cc.gatech.edu/~judy/domainadapt/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    paths = dl_manager.download_and_extract({'train': self.DATA_URL})
    return {
        'train': self._generate_examples(
            data_path=paths['train'] / self.builder_config.name / 'images'
        ),
    }

  def _generate_examples(self, data_path):
    counter = itertools.count()

    for path in data_path.iterdir():
      for image_path in path.iterdir():
        yield next(counter), {
            'image': image_path,
            'label': path.name,
        }
