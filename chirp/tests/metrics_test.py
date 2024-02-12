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

"""Tests for metrics."""

import functools
import os
from chirp.models import cwt
from chirp.models import metrics
from clu import metrics as clu_metrics
import flax
import jax
from jax import numpy as jnp
from absl.testing import absltest


@flax.struct.dataclass
class ValidationMetrics(clu_metrics.Collection):
  valid_map: clu_metrics.Average.from_fun(metrics.map_)


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
              logits=logits, labels=labels
          )
      )

    @functools.partial(jax.pmap, axis_name="batch")
    def p_update_metrics(valid_metrics, labels, logits):
      return valid_metrics.merge(
          ValidationMetrics.gather_from_model_output(
              logits=logits, labels=labels, axis_name="batch"
          )
      )

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

    for k in ["valid_map"]:
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
    self.assertEqual(av_prec, 0.0)

  def test_least_squares_mixit(self):
    # Create some genuinely interesting source signals...
    xs = jnp.linspace(-jnp.pi, jnp.pi, 256)
    f3 = cwt.gabor_filter(3, cwt.Domain.TIME, cwt.Normalization.L2)(xs).real
    f9 = cwt.gabor_filter(9, cwt.Domain.TIME, cwt.Normalization.L2)(xs).real
    f5 = cwt.gabor_filter(5, cwt.Domain.TIME, cwt.Normalization.L2)(xs).real
    f25 = cwt.gabor_filter(25, cwt.Domain.TIME, cwt.Normalization.L2)(xs).real

    mix1 = f3 + f9
    mix2 = f5 + f25
    reference = jnp.concatenate(
        [mix1[jnp.newaxis, jnp.newaxis, :], mix2[jnp.newaxis, jnp.newaxis, :]],
        1,
    )
    estimate = jnp.concatenate(
        [
            f3[jnp.newaxis, jnp.newaxis, :],
            f5[jnp.newaxis, jnp.newaxis, :],
            f9[jnp.newaxis, jnp.newaxis, :],
            f25[jnp.newaxis, jnp.newaxis, :],
        ],
        1,
    )
    best_mix, mix_matrix = metrics.least_squares_mixit(reference, estimate)

    l1_err = lambda x, y: jnp.sum(jnp.abs(x - y))
    # The mix matrix corresponding to the definition of mix1 and mix2.
    expected_mix = jnp.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    self.assertEqual(l1_err(mix_matrix, expected_mix), 0.0)

    # The best_mix should recover the mixture channels exactly.
    self.assertEqual(l1_err(best_mix[0, 0], mix1), 0.0)
    self.assertEqual(l1_err(best_mix[0, 1], mix2), 0.0)


if __name__ == "__main__":
  absltest.main()
