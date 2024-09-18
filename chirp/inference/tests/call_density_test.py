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

"""Tests for call density estimation."""

import collections
import shutil
import string
import tempfile

from chirp.inference import call_density
from etils import epath
import numpy as np
from sklearn import metrics

from absl.testing import absltest


class CallDensityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def generate_scores(self, samples_per_bin, quantile_bounds):
    bin_weights = quantile_bounds[1:] - quantile_bounds[:-1]
    min_weight = np.min(bin_weights)

    # We need to generate enough samples to ensure at least 'samples_per_bin'
    # examples in each bin.
    num_samples = (int(2.0 / min_weight) + 1) * samples_per_bin
    return np.random.normal(size=num_samples)

  def generate_data(
      self,
      samples_per_bin: int,
      quantile_bounds: np.ndarray,
      pos_rates: np.ndarray | None = None,
      seed: int = 42,
      scores: np.ndarray | None = None,
      labels: np.ndarray | None = None,
  ):
    np.random.seed(seed)
    examples = []
    if scores is None:
      scores = self.generate_scores(samples_per_bin, quantile_bounds)

    value_bounds = np.quantile(scores, quantile_bounds)
    bin_weights = quantile_bounds[1:] - quantile_bounds[:-1]
    binned_score_counts = collections.defaultdict(int)
    for i, s in enumerate(scores):
      result_bin = max(np.argmax(s < value_bounds) - 1, 0)
      if binned_score_counts[result_bin] >= samples_per_bin:
        continue
      binned_score_counts[result_bin] += 1
      if labels is None and pos_rates is not None:
        lbl = 2 * int(np.random.uniform(0.0, 1.0) < pos_rates[result_bin]) - 1
      elif labels is not None:
        lbl = labels[i]
      else:
        raise ValueError('Must provide either labels or pos_rates.')

      chars = np.array(list(string.ascii_letters + string.digits))
      random_filename = ''.join(np.random.choice(chars, 10))
      ex = call_density.ValidationExample(
          filename=random_filename,
          timestamp_offset=0.0,
          score=s,
          is_pos=lbl,
          bin=result_bin,
          bin_weight=bin_weights[result_bin],
      )
      examples.append(ex)
    return examples

  def test_estimate_call_density(self):
    quantile_bounds = np.array([0.0, 0.5, 0.75, 0.825, 1.0])
    pos_rates = np.array([0.01, 0.1, 0.4, 0.9])
    examples = self.generate_data(256, quantile_bounds, pos_rates)
    density_ev, density_samples = call_density.estimate_call_density(
        examples, 10_000
    )

    bin_weights = quantile_bounds[1:] - quantile_bounds[:-1]
    gt_estimate = np.dot(bin_weights, pos_rates)
    # For seeded data, we have gt 0.2175, ev 0.21335, sample mean 2.1333.
    self.assertAlmostEqual(density_ev, gt_estimate, places=1)
    self.assertAlmostEqual(density_ev, np.mean(density_samples), places=2)

  def test_estimate_roc_auc(self):
    samples_per_bin = 256
    quantile_bounds = np.array([0.0, 0.5, 0.75, 0.825, 1.0])
    bin_weights = quantile_bounds[1:] - quantile_bounds[:-1]
    min_weight = np.min(bin_weights)
    num_samples = (int(2.0 / min_weight) + 1) * samples_per_bin
    np.random.seed(42)

    # Generate some scores.
    noise_mu = 0.5
    noise_scores = np.random.normal(size=num_samples)
    labels = np.random.randint(0, 2, size=num_samples)
    scores = noise_mu * noise_scores + (1 - noise_mu) * labels
    gt_roc_auc = metrics.roc_auc_score(labels, scores)

    # Generate some validation examples.
    quantile_bounds = np.array([0.0, 0.5, 0.75, 0.825, 1.0])
    pos_rates = np.array([0.01, 0.1, 0.4, 0.9])
    examples = self.generate_data(
        samples_per_bin,
        quantile_bounds,
        pos_rates,
        seed=42,
        scores=scores,
        labels=2 * labels - 1,
    )
    roc_auc = call_density.estimate_roc_auc(examples)
    self.assertAlmostEqual(roc_auc, gt_roc_auc, places=1)

  def test_write_read_log(self):
    quantile_bounds = np.array([0.0, 0.5, 0.75, 0.825, 1.0])
    pos_rates = np.array([0.01, 0.1, 0.4, 0.9])
    examples = self.generate_data(256, quantile_bounds, pos_rates)
    log_filepath = call_density.write_validation_log(
        examples, epath.Path(self.tempdir), 'someclass'
    )
    got_examples = call_density.load_validation_log(log_filepath)
    self.assertLen(got_examples, len(examples))
    for ex, got_ex in zip(examples, got_examples):
      self.assertEqual(ex.filename, got_ex.filename)
      self.assertEqual(ex.timestamp_offset, got_ex.timestamp_offset)
      self.assertAlmostEqual(ex.score, got_ex.score)
      self.assertEqual(ex.is_pos, got_ex.is_pos)
      self.assertEqual(ex.bin, got_ex.bin)
      self.assertAlmostEqual(ex.bin_weight, got_ex.bin_weight)

    with self.subTest('idempotence'):
      log_filepath = call_density.write_validation_log(
          examples, epath.Path(self.tempdir), 'someclass'
      )
      got_examples = call_density.load_validation_log(log_filepath)
      self.assertLen(got_examples, len(examples))


if __name__ == '__main__':
  absltest.main()
