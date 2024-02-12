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

"""Training loop for MAE."""
import functools
from chirp.data import utils as data_utils
from chirp.models import mae
from chirp.models import taxonomy_model
from chirp.taxonomy import class_utils
from chirp.train import classifier
from chirp.train import train_utils
from clu import checkpoint
from clu import metric_writers
from clu import periodic_actions
import flax.jax_utils as flax_utils
import jax
from jax import numpy as jnp
from jax import random
from ml_collections import config_dict
import optax


def initialize_model(
    model_config: config_dict.ConfigDict,
    rng_seed: int,
    input_shape: tuple[int, ...],
    learning_rate: float,
    workdir: str,
) -> tuple[train_utils.ModelBundle, train_utils.TrainState]:
  """Creates model for training, eval, or inference."""
  del model_config
  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Handle lazy computation
  input_shape = tuple(s.get() if hasattr(s, "get") else s for s in input_shape)

  # Load model
  model_init_key, key = random.split(key)
  model = mae.MaskedAutoencoder(
      encoder=mae.Encoder(), decoder=mae.Decoder(output_size=input_shape)
  )
  variables = model.init(
      model_init_key, jnp.zeros((1,) + input_shape), train=False
  )
  model_state, params = variables.pop("params")
  # NOTE: https://github.com/deepmind/optax/issues/160
  params = params.unfreeze()

  # Initialize optimizer and handle constraints
  optimizer = optax.adamw(
      learning_rate=optax.cosine_decay_schedule(
          init_value=2 * learning_rate,
          # Assume 50 epochs with batches of 64
          decay_steps=2_914_000 * 50 // 64,
          alpha=1e-2,
      ),
      b2=0.95,
  )
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = train_utils.TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state
  )
  return (
      train_utils.ModelBundle(
          model=model, key=key, ckpt=ckpt, optimizer=optimizer
      ),
      train_state,
  )


def initialize_finetune_model(
    model_config: config_dict.ConfigDict,
    rng_seed: int,
    input_shape: tuple[int, ...],
    learning_rate: float,
    workdir: str,
    target_class_list: str,
) -> tuple[
    classifier.train_utils.ModelBundle, classifier.train_utils.TrainState
]:
  """Creates model for training, eval, or inference."""
  # Initialize random number generator
  key = random.PRNGKey(rng_seed)

  # Handle lazy computation
  input_shape = tuple(s.get() if hasattr(s, "get") else s for s in input_shape)
  class_lists = class_utils.get_class_lists(target_class_list, True)

  # Load model
  model_init_key, key = random.split(key)
  model = taxonomy_model.TaxonomyModel(
      num_classes={k: len(v.classes) for (k, v) in class_lists.items()},
      encoder=mae.Embedder(encoder=mae.Encoder(mask_rate=0.75)),
      taxonomy_loss_weight=0.0,
  )
  variables = model.init(
      model_init_key, jnp.zeros((1,) + input_shape), train=False
  )
  model_state, params = variables.pop("params")
  # NOTE: https://github.com/deepmind/optax/issues/160
  params = params.unfreeze()

  # Load checkpoint
  mae_model_bundle, mae_train_state = initialize_model(
      **model_config.mae_init_config
  )
  mae_train_state = mae_model_bundle.ckpt.restore(mae_train_state)
  params["encoder"]["encoder"] = mae_train_state.params["encoder"]
  if mae_train_state.model_state:
    raise ValueError(
        "currently only models without model state "
        "(such as batch statistics) are handled"
    )

  # Initialize optimizer and handle constraints
  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  # Load checkpoint
  ckpt = checkpoint.MultihostCheckpoint(workdir)
  train_state = classifier.train_utils.TrainState(
      step=0, params=params, opt_state=opt_state, model_state=model_state
  )
  return (
      classifier.train_utils.ModelBundle(
          model=model,
          key=key,
          ckpt=ckpt,
          optimizer=optimizer,
          class_lists=class_lists,
      ),
      train_state,
  )


def train(
    model_bundle,
    train_state,
    train_dataset,
    num_train_steps: int,
    logdir: str,
    log_every_steps: int,
    checkpoint_every_steps: int,
) -> None:
  """Train a model.

  Args:
    model_bundle: Static objects for conducting the experiment.
    train_state: Initial train_utils.TrainState.
    train_dataset: Training dataset.
    num_train_steps: The number of training steps.
    logdir: Directory to use for logging.
    log_every_steps: Write the training minibatch loss.
    checkpoint_every_steps: Checkpoint the model and training state.
  """
  train_iterator = train_dataset.as_numpy_iterator()

  # Forward pass and metrics
  def forward(params, key, batch, model_state):
    dropout_key, patch_mask_key = random.split(key)
    variables = {"params": params, **model_state}
    model_outputs, model_state = model_bundle.model.apply(
        variables,
        batch["audio"],
        train=True,
        mutable=list(model_state.keys()),
        rngs={"dropout": dropout_key, "patch_mask": patch_mask_key},
    )
    # The decoded patches, the original patches, and indices of the ones that
    # were masked
    decoded_patches, patches, masked = model_outputs
    loss = (
        jnp.mean(
            jnp.sum(
                (
                    jnp.take_along_axis(
                        patches, masked[..., jnp.newaxis], axis=1
                    )
                    - jnp.take_along_axis(
                        decoded_patches, masked[..., jnp.newaxis], axis=1
                    )
                )
                ** 2,
                axis=-1,
            )
        )
        / model_bundle.model.encoder.mask_rate
    )
    b, h, w, c = batch["audio"].shape
    ph, pw = model_bundle.model.encoder.patch_size
    reconstructed = jnp.reshape(
        decoded_patches, (b, h // ph, w // pw, ph, pw, c)
    )
    reconstructed = jnp.reshape(
        jnp.swapaxes(reconstructed, -3, -4), (b, h, w, c)
    )
    images = {
        "original": batch["audio"][:1],
        "reconstructed": reconstructed[:1],
    }
    return loss, ({"loss": loss, "images": images}, model_state)

  # Define update step
  @functools.partial(jax.pmap, axis_name="batch")
  def update_step(key, batch, train_state):
    grads, (train_metrics, model_state) = jax.grad(forward, has_aux=True)(
        train_state.params, key, batch, train_state.model_state
    )
    grads = jax.lax.pmean(grads, axis_name="batch")
    updates, opt_state = model_bundle.optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    params = optax.apply_updates(train_state.params, updates)
    train_state = train_utils.TrainState(
        step=train_state.step + 1,
        params=params,
        opt_state=opt_state,
        model_state=model_state,
    )
    return train_metrics, train_state

  initial_step = int(train_state.step)
  train_state = flax_utils.replicate(train_state)

  # Logging
  writer = metric_writers.create_default_writer(logdir)
  reporter = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )

  # Training and evaluation loop
  key = model_bundle.key

  for step in range(initial_step, num_train_steps + 1):
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      batch = next(train_iterator)
      step_key, key = random.split(key)
      step_key = random.split(step_key, num=jax.local_device_count())
      train_metrics, train_state = update_step(step_key, batch, train_state)
      train_metrics = flax_utils.unreplicate(train_metrics)

      if step % log_every_steps == 0:
        images = train_metrics.pop("images")
        writer.write_scalars(step, train_metrics)
        writer.write_summaries(step, images)
      reporter(step)

    if (step + 1) % checkpoint_every_steps == 0 or step == num_train_steps:
      with reporter.timed("checkpoint"):
        model_bundle.ckpt.save(flax_utils.unreplicate(train_state))
  writer.close()


def run(
    mode: str,
    config: config_dict.ConfigDict,
    workdir: str,
    tf_data_service_address: str,
) -> None:
  """Run the experiment."""
  train_dataset, valid_dataset, dataset_info = None, None, None
  if mode in ("train", "finetune"):
    train_dataset, dataset_info = data_utils.get_dataset(
        is_train=True,
        tf_data_service_address=tf_data_service_address,
        **config.train_dataset_config,
    )
  elif mode == "eval":
    valid_dataset, dataset_info = data_utils.get_dataset(
        **config.eval_dataset_config
    )
  elif mode == "export":
    valid_dataset, dataset_info = None, None

  if (
      dataset_info is not None
      and dataset_info.features["audio"].sample_rate != config.sample_rate_hz
  ):
    raise ValueError(
        "Dataset sample rate must match config sample rate. To address this, "
        "need to set the sample rate in the config to {}.".format(
            dataset_info.features["audio"].sample_rate
        )
    )

  if mode == "train":
    model_bundle, train_state = initialize_model(
        workdir=workdir, **config.init_config
    )
  else:
    model_bundle, train_state = initialize_finetune_model(
        workdir=workdir, **config.init_config
    )
  if mode == "train":
    train_state = model_bundle.ckpt.restore_or_initialize(train_state)
    train(
        model_bundle,
        train_state,
        train_dataset,
        logdir=workdir,
        **config.train_config,
    )
  if mode == "finetune":
    train_state = model_bundle.ckpt.restore_or_initialize(train_state)
    classifier.train(
        model_bundle,
        train_state,
        train_dataset,
        logdir=workdir,
        **config.train_config,
    )
  elif mode == "eval":
    classifier.evaluate(
        model_bundle,
        train_state,
        valid_dataset,
        loss_fn=config.loss_fn,
        workdir=workdir,
        **config.eval_config,
    )
  elif mode == "export":
    classifier.export_tf_model(
        model_bundle,
        train_state,
        workdir=workdir,
        **config.export_config,
    )
