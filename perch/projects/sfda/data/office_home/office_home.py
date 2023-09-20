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

"""Office-Home dataset."""

import itertools

import tensorflow_datasets as tfds

_DESCRIPTION = """
Office-Home dataset.
"""

_CITATION = """
@inproceedings{venkateswara2017deep,
  title={Deep hashing network for unsupervised domain adaptation},
  author={Venkateswara, Hemanth and Eusebio, Jose and Chakraborty, Shayok and Panchanathan, Sethuraman},
  booktitle={Proceedings of the Conference on Computer Vision and Pattern Recognition},
  pages={5018--5027},
  year={2017}
}
"""

_MANUAL_DOWNLOAD_INSTRUCTIONS = """
Download and unzip OfficeHomeDataset_10072016.zip from
https://drive.google.com/uc?export=download&id=0B81rNlvomiwed0V1YUxQdC1uOTg,
then move the OfficeHomeDataset_10072016/ directory into TFDS' manual download
directory.
"""


class OfficeHome(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for the Office-Home dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = _MANUAL_DOWNLOAD_INSTRUCTIONS
  BUILDER_CONFIGS = [
      tfds.core.BuilderConfig(name=name)
      for name in ('art', 'clipart', 'product', 'real_world')
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'label': tfds.features.ClassLabel(
                names=[
                    'alarm_clock',
                    'backpack',
                    'batteries',
                    'bed',
                    'bike',
                    'bottle',
                    'bucket',
                    'calculator',
                    'calendar',
                    'candles',
                    'chair',
                    'clipboards',
                    'computer',
                    'couch',
                    'curtains',
                    'desk_lamp',
                    'drill',
                    'eraser',
                    'exit_sign',
                    'fan',
                    'file_cabinet',
                    'flipflops',
                    'flowers',
                    'folder',
                    'fork',
                    'glasses',
                    'hammer',
                    'helmet',
                    'kettle',
                    'keyboard',
                    'knives',
                    'lamp_shade',
                    'laptop',
                    'marker',
                    'monitor',
                    'mop',
                    'mouse',
                    'mug',
                    'notebook',
                    'oven',
                    'pan',
                    'paper_clip',
                    'pen',
                    'pencil',
                    'postit_notes',
                    'printer',
                    'push_pin',
                    'radio',
                    'refrigerator',
                    'ruler',
                    'scissors',
                    'screwdriver',
                    'shelf',
                    'sink',
                    'sneakers',
                    'soda',
                    'speaker',
                    'spoon',
                    'table',
                    'telephone',
                    'toothbrush',
                    'toys',
                    'trash_can',
                    'tv',
                    'webcam',
                ]
            ),
        }),
        supervised_keys=('image', 'label'),
        homepage='https://www.hemanthdv.org/officeHomeDataset.html',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    domain = {
        'art': 'Art',
        'clipart': 'Clipart',
        'product': 'Product',
        'real_world': 'Real World',
    }[self.builder_config.name]
    return {
        'train': self._generate_examples(
            data_path=dl_manager.manual_dir
            / 'OfficeHomeDataset_10072016'
            / domain
        ),
    }

  def _generate_examples(self, data_path):
    counter = itertools.count()

    for path in data_path.iterdir():
      for image_path in path.iterdir():
        yield next(counter), {
            'image': image_path,
            'label': path.name.lower(),
        }
