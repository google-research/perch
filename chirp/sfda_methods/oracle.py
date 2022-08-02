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

"""An oracle that trains on the labels provided in the target domain."""

import functools
from typing import Dict

from chirp import adapt
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax.jax_utils as flax_utils
import jax
import numpy as np
import optax
from tensorflow import data

ModelBundle = adapt.ModelBundle
AdaptationState = adapt.AdaptationState


class Oracle(adapt.SFDAMethod):
  """An oracle that trains on the labels provided in the target domain.

  This oracle will be used as an upper bound for SFDA methods.
  """

  def do_epoch(self, key: jax.random.PRNGKeyArray,
               model_bundles: Dict[str, ModelBundle],
               adaptation_state: AdaptationState,
               adaptation_dataset: data.Dataset,
               metrics_collection: clu_metrics.Collection,
               writer: metric_writers.MetricWriter,
               reporter: periodic_actions.ReportProgress) -> AdaptationState:

    # Classification forward pass and metrics
    def main_forward(params, key, batch, model_state):
      dropout_key, low_pass_key = jax.random.split(key)
      variables = {"params": params, **model_state}
      model_outputs, model_state = model_bundles["main"].model.apply(
          variables,
          batch["audio"],
          train=True,
          mutable=list(model_state.keys()),
          rngs={
              "dropout": dropout_key,
              "low_pass": low_pass_key
          })
      batch_metrics = metrics_collection.gather_from_model_output(
          outputs=model_outputs,
          label=batch["label"],
          genus=batch["genus"],
          family=batch["family"],
          order=batch["order"]).compute()
      return batch_metrics["adaptation___supervised_loss"], (batch_metrics,
                                                             model_state)

    @functools.partial(jax.pmap, axis_name="batch")
    def update_step(batch, adaptation_state, step_key):

      # Get main loss' gradients.
      params = adaptation_state.models_params["main"]
      main_model_state = adaptation_state.model_states["main"]
      main_opt_state = adaptation_state.opt_states["main"]
      grads, (batch_metrics, main_model_state) = jax.grad(
          main_forward, has_aux=True)(params, step_key, batch, main_model_state)

      # All gather gradients and average them.
      grads = jax.lax.pmean(grads, axis_name="batch")

      # Get optimizer updates. While doing this, get new opt_state.
      updates, main_opt_state = model_bundles["main"].optimizer.update(
          grads, main_opt_state, params)

      # Apply those updates to get the new model's state.
      params = optax.apply_updates(params, updates)
      adaptation_state = AdaptationState(
          step=adaptation_state.step + 1,
          epoch=adaptation_state.epoch + 1,
          models_params={"main": params},
          opt_states={"main": main_opt_state},
          model_states={"main": main_model_state})
      return batch_metrics, adaptation_state

    # Replicate everything across devices
    adaptation_state = flax_utils.replicate(adaptation_state)

    for step, batch in enumerate(adaptation_dataset.as_numpy_iterator()):

      step_key, key = jax.random.split(key)
      step_key = jax.random.split(step_key, num=jax.local_device_count())
      batch = {
          k: v
          for k, v in batch.items()
          if k in ["audio", "label", "genus", "family", "order"]
      }
      batch = jax.tree_map(np.asarray, batch)
      batch_metrics, adaptation_state = update_step(batch, adaptation_state,
                                                    step_key)
      # step = flax_utils.unreplicate(adaptation_state).step[0]
      reporter(step)
      writer.write_scalars(step, flax_utils.unreplicate(batch_metrics))

    return flax_utils.unreplicate(adaptation_state)
