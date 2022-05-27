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

"""Tests for metrics."""

import functools
import os
from chirp.models import class_average
from chirp.models import metrics
from clu import metrics as clu_metrics
import flax
import jax
from absl.testing import absltest


@flax.struct.dataclass
class ValidationMetrics(clu_metrics.Collection):
  valid_map: clu_metrics.Average.from_fun(metrics.map_)
  valid_cmap: class_average.ClassAverage.from_fun(metrics.cmap)


class MetricsTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Test with two CPU devices.
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

  def test_parallel_metric_agreemenet(self):

    @jax.jit
    def update_metrics(valid_metrics, labels, logits):
      return valid_metrics.merge(
          ValidationMetrics.single_from_model_output(
              logits=logits, labels=labels))

    @functools.partial(jax.pmap, axis_name="batch")
    def p_update_metrics(valid_metrics, labels, logits):
      return valid_metrics.merge(
          ValidationMetrics.gather_from_model_output(
              logits=logits, labels=labels, axis_name="batch"))

    batch_size = 4
    num_classes = 5
    key = jax.random.PRNGKey(2)

    logits = jax.random.normal(key, [batch_size, num_classes])
    labels = jax.numpy.float32(logits < 0)

    valid_metrics = ValidationMetrics.empty()
    valid_metrics = update_metrics(valid_metrics, labels, logits)
    serial_metrics = valid_metrics.compute()

    # Compute replicated metrics.
    valid_metrics = flax.jax_utils.replicate(ValidationMetrics.empty())
    logits_repl = flax.jax_utils.replicate(logits)
    labels_repl = flax.jax_utils.replicate(labels)
    valid_metrics = p_update_metrics(valid_metrics, labels_repl, logits_repl)
    repl_metrics = flax.jax_utils.unreplicate(valid_metrics).compute()

    for k in ["valid_map", "valid_cmap"]:
      self.assertEqual(serial_metrics[k], repl_metrics[k])
      self.assertGreaterEqual(serial_metrics[k], 0.0)
      self.assertLessEqual(serial_metrics[k], 1.0)

  def test_average_precision_no_labels(self):
    batch_size = 4
    num_classes = 5
    key = jax.random.PRNGKey(2)

    logits = jax.random.normal(key, [batch_size, num_classes])
    labels = jax.numpy.zeros_like(logits)
    av_prec = jax.numpy.mean(metrics.average_precision(logits, labels))
    self.assertEqual(av_prec, 1.0)


if __name__ == "__main__":
  absltest.main()
