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

"""Tests for colab display elements."""

import os
import shutil
import tempfile
from unittest import mock

from chirp import audio_utils
from chirp.projects.agile2 import embedding_display
from chirp.projects.agile2.tests import test_utils
from chirp.projects.hoplite import interface
import IPython
import ipywidgets
import numpy as np

from absl.testing import absltest


class EmbeddingDisplayTest(absltest.TestCase):

  def setUp(self):
    # Without this, unit tests using Ipywidgets will fail with 'Comms cannot be
    # opened without a kernel and a comm_manager attached to that kernel'. This
    # mocks out the comms. This is a little fragile because it sets a private
    # attribute and may break for future Ipywidget library upgrades.
    setattr(
        ipywidgets.Widget,
        '_comm_default',
        lambda self: mock.MagicMock(spec=IPython.kernel.comm.Comm),
    )

    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def test_query_display(self):
    classes = ['pos', 'neg']
    filenames = ['foo', 'bar', 'baz']
    sample_rate_hz = 16000
    test_utils.make_wav_files(
        self.tempdir,
        classes,
        filenames,
        file_len_s=6.0,
        sample_rate_hz=sample_rate_hz,
    )

    qd = embedding_display.QueryDisplay(
        uri=os.path.join(self.tempdir, 'pos', 'foo_pos.wav'),
        window_size_s=3.0,
        sample_rate_hz=sample_rate_hz,
    )
    # Check that we can load the audio file, and obtain the entire file.
    qd.update_audio()
    self.assertLen(qd.audio, 16000 * 6)

    qd.update_spectrogram()
    self.assertEqual(qd.full_spectrogram.shape, (600, 96))
    self.assertEqual(qd.window_spectrogram.shape, (300, 96))
    initial_spectrogram = qd.window_spectrogram.copy()

    qd.offset_s = 3.0
    qd.update_spectrogram()
    self.assertEqual(qd.window_spectrogram.shape, (300, 96))
    # Check that the spectrogram changed.
    diff = np.abs(initial_spectrogram - qd.window_spectrogram).sum()
    self.assertGreater(diff, 1.0)

    qd.offset_s = 0.0
    qd.update_spectrogram()
    self.assertEqual(qd.window_spectrogram.shape, (300, 96))
    # Check that the spectrogram is back to the original.
    diff = np.abs(initial_spectrogram - qd.window_spectrogram).sum()
    self.assertEqual(diff, 0.0)

  def test_embedding_display_button(self):
    disp = embedding_display.EmbeddingDisplay(
        embedding_id=123,
        dataset_name='test_dataset',
        uri='test_uri',
        offset_s=1.0,
        score=0.5,
    )
    disp._make_label_widgets(('foo', 'bar'))
    foo_button = disp.widgets['foo']
    bar_button = disp.widgets['bar']
    self.assertIsNotNone(foo_button)
    self.assertIsNotNone(bar_button)
    # Buttons start with value 0, indicating no label.
    # Clicking each button cycles 0 -> 1 -> -1 -> 0.
    self.assertEqual(foo_button.value, 0)
    self.assertEqual(bar_button.value, 0)
    self.assertEmpty(disp.harvest_labels('test_provenance'))
    foo_button.click()
    self.assertEqual(foo_button.value, 1)
    self.assertEqual(bar_button.value, 0)
    bar_button.click()
    self.assertEqual(foo_button.value, 1)
    self.assertEqual(bar_button.value, 1)
    foo_button.click()
    self.assertEqual(foo_button.value, -1)
    self.assertEqual(bar_button.value, 1)
    labels = disp.harvest_labels('test_provenance')
    self.assertLen(labels, 2)
    self.assertEqual(labels[0].embedding_id, 123)
    self.assertEqual(labels[0].label, 'foo')
    self.assertEqual(labels[0].type, interface.LabelType.NEGATIVE)
    self.assertEqual(labels[0].provenance, 'test_provenance')

    self.assertEqual(labels[1].embedding_id, 123)
    self.assertEqual(labels[1].label, 'bar')
    self.assertEqual(labels[1].type, interface.LabelType.POSITIVE)
    self.assertEqual(labels[1].provenance, 'test_provenance')

    bar_button.click()
    self.assertEqual(foo_button.value, -1)
    self.assertEqual(bar_button.value, -1)
    foo_button.click()
    self.assertEqual(foo_button.value, 0)
    self.assertEqual(bar_button.value, -1)
    bar_button.click()
    self.assertEqual(foo_button.value, 0)
    self.assertEqual(bar_button.value, 0)

  def test_embedding_display_group(self):
    classes = ['pos', 'neg']
    filenames = ['foo', 'bar', 'baz']
    sample_rate_hz = 16000
    test_utils.make_wav_files(
        self.tempdir,
        classes,
        filenames,
        file_len_s=6.0,
        sample_rate_hz=sample_rate_hz,
    )
    member0 = embedding_display.EmbeddingDisplay(
        embedding_id=123,
        dataset_name='test_dataset',
        uri=os.path.join(self.tempdir, 'pos', 'foo_pos.wav'),
        offset_s=1.0,
        score=0.5,
        sample_rate_hz=sample_rate_hz,
    )
    member1 = embedding_display.EmbeddingDisplay(
        embedding_id=456,
        dataset_name='test_dataset',
        uri=os.path.join(self.tempdir, 'neg', 'bar_neg.wav'),
        offset_s=2.0,
        score=0.6,
        sample_rate_hz=sample_rate_hz,
    )

    audio_loader = lambda uri, offset_s: audio_utils.load_audio_window(
        uri, offset_s=offset_s, sample_rate=sample_rate_hz, window_size_s=3.0
    )
    group = embedding_display.EmbeddingDisplayGroup.create(
        members=[member0, member1],
        sample_rate_hz=sample_rate_hz,
        audio_loader=audio_loader,
    )
    self.assertLen(group.members, 2)
    self.assertEqual(group.members[0].embedding_id, 123)
    self.assertEqual(group.members[1].embedding_id, 456)
    self.assertEqual(group.members[0].dataset_name, 'test_dataset')
    for got_member in group.iterator_with_audio():
      self.assertIsNotNone(got_member.audio)
      self.assertIsNotNone(got_member.spectrogram)


if __name__ == '__main__':
  absltest.main()
