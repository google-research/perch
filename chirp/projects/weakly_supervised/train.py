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

"""Weakly supervised learning model."""
import functools
from typing import Any

from chirp.projects.weakly_supervised import data
from chirp.projects.weakly_supervised import model
from chirp.projects.weakly_supervised import validate
from chirp.taxonomy import namespace_db
from clu import metric_writers
from flax import core
from flax import jax_utils
from flax import struct
from flax.training import train_state
from grain import python as pygrain
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import tree_util
from ml_collections import config_dict
import optax
from orbax import checkpoint


class TrainState(train_state.TrainState):
  variables: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


def run(
    mode: str,
    config: config_dict.ConfigDict,
    workdir: str,
    tf_data_service_address: str,
) -> None:
  """Train a supervised contrastive model."""
  if tf_data_service_address:
    raise ValueError

  if mode == "train":
    # Load the model
    db = namespace_db.load_db()
    num_classes = len(db.class_lists[config.source_class_list].classes)
    instance_model = model.InstanceModel(num_classes=num_classes)
    key = random.PRNGKey(config.seed)
    init_key, key = random.split(key)
    variables = instance_model.init(
        {"params": init_key},
        jnp.ones((1, int(config.window_size * config.sample_rate))),
        train=False,
    )
    params = variables.pop("params")
    state = TrainState.create(
        apply_fn=instance_model.apply,
        params=params,
        variables=variables,
        tx=optax.adam(learning_rate=config.learning_rate),
    )

    def forward(params, variables, batch, key) -> tuple[jnp.ndarray, Any]:
      dropout_key, params_key = random.split(key)
      out, updated_variables = state.apply_fn(
          variables | {"params": params},
          batch["windows"],
          rngs={"dropout": dropout_key, "params": params_key},
          train=True,
          mutable=variables.keys(),
      )
      loss = model.weakly_supervised_sigmoid_binary_cross_entropy(
          out, batch["label"], jnp.ravel(batch["sentinel"]).astype(bool)
      )
      return loss, updated_variables

    grad_fn = jax.value_and_grad(forward, has_aux=True)

    @functools.partial(jax.pmap, axis_name="batch", in_axes=(0, 0, None))
    def step(state, batch, step_key):
      (loss, variables), grads = grad_fn(
          state.params,
          state.variables,
          batch,
          step_key,
      )
      # Calculate the total number of bags across all devices
      num_records = lax.psum(jnp.sum(batch["sentinel"]), axis_name="batch")
      # Note that each device might have had a different number of bags, and
      # more bags leads to a higher loss and larger gradient. So instead of just
      # averaging, we take the sum of all gradients and then normalize using the
      # total number of bags.
      grads = tree_util.tree_map(
          lambda x: x / num_records, lax.psum(grads, axis_name="batch")
      )
      # The batch normalization values are just averaged since the number of
      # examples seen by each device is nearly the same (not exactly, since
      # batches might have been padded, but this should be a minor difference).

      # Note that the padding doesn't affect the training (because they have no
      # loss) but they do affect the batch normalization running averages, i.e.,
      # those values aren't entirely correct. However, we are going to assume
      # that there is little enough padding that it doesn't really matter...
      variables = lax.pmean(variables, axis_name="batch")
      state = state.apply_gradients(grads=grads, variables=variables)
      # Just like gradients, the loss needs to be normalized with the total
      # number of bags.
      return lax.psum(loss, axis_name="batch") / num_records, state

    # Load the dataset
    data_loader = data.get_dataset(config)
    data_iter = iter(data_loader)

    # Load the validation set
    validation_data_loader = data.get_validation_dataset(config)

    # TODO(bartvm): This is likely a bit slow because audio consists of batches
    # of variable sizes (triggering recompilation), and is not pmapped.
    @jax.jit
    def embed_fn(params, variables, audio):
      return state.apply_fn(
          variables | {"params": params},
          audio,
          train=False,
          capture_intermediates=True,
      )[1]["intermediates"]["embeddings"][0]

    # Checkpointing
    checkpoint_manager = checkpoint.CheckpointManager(
        workdir,
        {
            "state": checkpoint.PyTreeCheckpointer(),
            "data_iter": checkpoint.Checkpointer(
                pygrain.PyGrainCheckpointHandler()  # pytype:disable=wrong-arg-types
            ),
        },
        options=checkpoint.CheckpointManagerOptions(
            max_to_keep=3, keep_period=2
        ),
    )

    step_num = 0
    if checkpoint_manager.latest_step() is not None:
      step_num = checkpoint_manager.latest_step()
      # restore_args is only for restoring in a multi-device/sharded setting
      restore_args = checkpoint.checkpoint_utils.construct_restore_args(
          state,
          jax.tree_map(
              lambda x: x.sharding if isinstance(x, jax.Array) else None, state
          ),
      )
      restored = checkpoint_manager.restore(
          checkpoint_manager.latest_step(),
          items={"state": state, "data_iter": data_iter},
          restore_kwargs={"state": {"restore_args": restore_args}},
      )
      state, data_iter = restored["state"], restored["data_iter"]

    state = jax_utils.replicate(state)

    # Metrics writing
    writer = metric_writers.create_default_writer(workdir)

    for batch in data_iter:
      with jax.profiler.StepTraceAnnotation("train", step_num=step_num):
        key, step_key = random.split(key)
        loss, state = step(
            state,
            {
                "windows": batch["windows"],
                "label": batch["label"],
                "sentinel": batch["sentinel"],
            },
            step_key,
        )
        if step_num % config.log_interval == 0:
          writer.write_scalars(step_num, {"loss": loss[0]})
        if step_num % config.validation_interval == 0:
          score = validate.one_shot_validate(
              random.PRNGKey(config.seed),
              validation_data_loader,
              functools.partial(
                  embed_fn,
                  jax_utils.unreplicate(state.params),
                  jax_utils.unreplicate(state.variables),
              ),
              config.num_one_shot_samples,
          )
          writer.write_scalars(step_num, {"validation_score": score})
        step_num += 1
        # TODO(bartvm): Figure out checkpointing with Orbax.
