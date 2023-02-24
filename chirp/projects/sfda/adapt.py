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

"""Adaptation and evaluation utilities for source-free domain adaptation."""

import abc
import enum
import functools
import time
from typing import Any, Callable

from absl import logging
from chirp.models import metrics as chirp_metrics
from chirp.models import output
from chirp.projects.sfda import losses
from chirp.projects.sfda import mca
from chirp.projects.sfda import metrics
from chirp.projects.sfda import model_utils
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
import flax
from flax import linen as nn
import flax.jax_utils as flax_utils
import jax
from jax import numpy as jnp
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow as tf
import tqdm


ForwardStepType = Callable[
    [
        dict[str, jnp.ndarray],
        flax.core.scope.FrozenVariableDict,
        flax.core.scope.VariableDict,
        jax.random.PRNGKeyArray | None,
    ],
    output.ClassifierOutput,
]


@flax.struct.dataclass
class AdaptationState:
  """All useful states and parameters kept during adaptation.

  Unlike in train.py where TrainState contains a single model, adaptation
  methods may use several methods (e.g. a teacher and a student, or an
  auxiliary GAN). Therefore, model_params, model_state and opts_states are
  stored in Dict in which the keys refer to the name of the model. The key
  'main' will be used for evaluation.

  Attributes:
    step: The batch iteration of adaptation (no reset after each epoch).
    epoch: The epoch of adaptation.
    model_params: The parameters of the model.
    model_state: The state of the model.
    method_state: All values a method needs to keep track of during adaptation.
    opt_state: The optimizer's state.
    restrict_classes: Whether to restrict to classes present in the target
      dataset.
  """

  step: int
  epoch: int
  model_params: flax.core.scope.VariableDict
  model_state: flax.core.scope.FrozenVariableDict
  method_state: dict[str, Any]
  opt_state: optax.OptState
  restrict_classes: bool


class Modality(enum.Enum):
  """Used to specify which modality we're using for adaptation."""

  IMAGE = "image"
  AUDIO = "audio"


def keep_jax_types(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """Remove non-numeric arrays from batch, to make it jit-compliant.

  Args:
    batch: The batch of data.

  Returns:
    The filtered batch of data, where any array containing non-numeric values
      has been filtered out.
  """
  return {k: v for k, v in batch.items() if v.dtype != np.dtype("O")}


class SFDAMethod(metaclass=abc.ABCMeta):
  """A template for a Source-Free Domain Adaptation (SFDA) method."""

  def initialize(
      self,
      model_config: config_dict.ConfigDict,
      rng_seed: int,
      modality: Modality,
      input_shape: tuple[int, ...],
      target_class_list: str,
      adaptation_iterations: int,
      optimizer_config: config_dict.ConfigDict,
      pretrained: bool,
  ) -> tuple[
      model_utils.ModelBundle,
      AdaptationState,
      jax.random.PRNGKeyArray,
      Callable[[Any, Any, str], Any],
      Callable[[Any], Any],
  ]:
    """Loads model's params and state, and instantiates the adaptation state.

    Args:
      model_config: The model configuration, including the definitions of the
        different parts of the architecture.
      rng_seed: The random seed used to define the jax random key and seed other
        non-jax random operations.
      modality: The modality currently used between 'image' and 'audio'.
      input_shape: The shape of the input.
      target_class_list: The classlist in which labels are expressed. Used to
        define the size of the classifier's head.
      adaptation_iterations: The total number of steps used for adaptation. Used
        to adequately define learning rate scheduling.
      optimizer_config: The optimizer configuration, including the name of the
        optimizer, the learning rate etc.
      pretrained: Whether to use a pretrained model or not.

    Returns:
      The model_bundle to use, the initial adaptation_state, and the jax
        random key to use for adaptation and the functions for renaming and
        reversing the renaming of parameters, needed in the case where different
        learning rates are used for different subsets of parameters.

    Raises:
      ValueError: In case the chosen modality is neither Modality.AUDIO
        nor Modality.IMAGE.
    """
    # Generate a random key
    key = jax.random.PRNGKey(rng_seed)
    if modality == Modality.AUDIO:
      prepare_fn = model_utils.prepare_audio_model
      restrict_classes = True
    elif modality == Modality.IMAGE:
      prepare_fn = model_utils.prepare_image_model
      restrict_classes = False
    else:
      raise ValueError(f"Modality {modality} not supported.")

    (
        model_bundle,
        params,
        model_state,
        opt_state,
        rename_fn,
        inverse_rename_fn,
    ) = prepare_fn(
        model_config=model_config,
        optimizer_config=optimizer_config,
        pretrained=pretrained,
        rng_seed=rng_seed,
        input_shape=input_shape,
        target_class_list=target_class_list,
        total_steps=adaptation_iterations,
    )

    # Package model, parameters and states in structures.
    # TODO(mboudiaf): Add support for restoring previous adaptation state.
    adaptation_state = AdaptationState(
        step=0,
        epoch=0,
        model_params=params,
        opt_state=opt_state,
        method_state={},
        model_state=model_state,
        restrict_classes=restrict_classes,
    )

    return model_bundle, adaptation_state, key, rename_fn, inverse_rename_fn

  @abc.abstractmethod
  def get_adaptation_metrics(
      self, supervised: bool, multi_label: bool, **_
  ) -> type[clu_metrics.Collection]:
    """Define metrics tracked during adaptation.

    On top of common metrics (accuracy/mAP ...), SFDA methods should
    specify the field 'main_loss', corresponding to the loss minimized during
    adaptation.

    Args:
      supervised: Whether the adaptation dataset is supervised. Used to
        determine if supervised metrics (e.g. accuracy) can be tracked or not.
      multi_label: Whether the current classification dataset is single-label or
        multi-label. Used to define appropriate metrics.
    """
    pass

  def before_run(
      self,
      key: jax.random.PRNGKeyArray,
      model_bundle: model_utils.ModelBundle,
      adaptation_state: AdaptationState,
      adaptation_dataset: tf.data.Dataset,
      modality: Modality,
      multi_label: bool,
      **method_kwargs,
  ) -> AdaptationState:
    """Any operation that a method needs to do before a run.

    An example of application is to initialize memories.

    Args:
      key: The jax random key used for random operations in this epoch.
      model_bundle: The ModelBundle used for adaptation.
      adaptation_state: The current state of adaptation.
      adaptation_dataset: The dataset used for adaptation.
      modality: The current modality.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Additional method-specific kwargs.

    Returns:
      A potentially updated adaptation_state.
    """
    del (
        key,
        model_bundle,
        modality,
        multi_label,
        method_kwargs,
        adaptation_dataset,
    )
    return adaptation_state

  def before_epoch(
      self,
      key: jax.random.PRNGKeyArray,
      model_bundle: model_utils.ModelBundle,
      adaptation_state: AdaptationState,
      adaptation_dataset: tf.data.Dataset,
      modality: Modality,
      multi_label: bool,
      **method_kwargs,
  ) -> AdaptationState:
    """Any operation that a method needs to do before an epoch.

    An example of application is to compute all pseudo-labels for the next
    epoch.

    Args:
      key: The jax random key used for random operations in this epoch.
      model_bundle: The ModelBundle used for adaptation.
      adaptation_state: The current state of adaptation.
      adaptation_dataset: The dataset used for adaptation.
      modality: The modality.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Additional method-specific kwargs.

    Returns:
      The adaptation state, with a potentially updated 'method_state' attribute.
    """
    del (
        key,
        model_bundle,
        adaptation_dataset,
        modality,
        multi_label,
        method_kwargs,
    )
    return adaptation_state

  # Prevent the forward_step from recompiling every step.
  def cache_get_forward_step(
      self,
      model: nn.Module,
      modality: Modality,
      use_batch_statistics: bool,
      train: bool = False,
  ) -> ForwardStepType:
    """Cache the forward step."""
    if hasattr(self, "_forward_step") and self._forward_step is not None:
      return self._forward_step
    self._forward_step = batch_forward(
        model, modality, use_batch_statistics, train
    )
    return self._forward_step

  # Prevent the update_step from recompiling every time do_epoch is called.
  def cache_get_update_step(self, update_step):
    """Cache the update step."""
    if hasattr(self, "_update_step") and self._update_step is not None:
      return self._update_step
    self._update_step = update_step
    return self._update_step

  # Prevent the update_step from recompiling every time do_epoch is called.
  def cache_get_update_metrics(self, update_metrics):
    """Cache the update step."""
    if hasattr(self, "_update_metrics") and self._update_metrics is not None:
      return self._update_metrics
    self._update_metrics = update_metrics
    return self._update_metrics

  def before_iter(
      self,
      key: jax.random.PRNGKeyArray,
      model_bundle: model_utils.ModelBundle,
      adaptation_state: AdaptationState,
      batch: dict[str, np.ndarray],
      modality: Modality,
      multi_label: bool,
      **method_kwargs,
  ) -> tuple[AdaptationState, dict[str, jnp.ndarray]]:
    """Any operation that a method needs to do before an adaptation iteration.

    An example of application is to compute the pseudo-labels needed for the
    current iteration.

    Args:
      key: The jax random key used for random operations in this epoch.
      model_bundle: The ModelBundle used for adaptation.
      adaptation_state: The current state of adaptation.
      batch: The iteration's batch of data.
      modality: The modality.
      multi_label: Whether this is a multi-label problem.
      **method_kwargs: Additional method-specific kwargs.

    Returns:
      A potentially updated version of the adaptation state
      A dictionary containing any variable the method may need during the next
        epoch of adaptation. All values in this dictionnary must be
        numeric-typed (they will be passed to a jitted function).
    """
    del (key, model_bundle, batch, modality, multi_label, method_kwargs)
    return adaptation_state, {}

  def do_epoch(
      self,
      key: jax.random.PRNGKeyArray,
      model_bundle: model_utils.ModelBundle,
      adaptation_state: AdaptationState,
      rename_fn: Callable[[Any, Any, str], Any],
      inverse_rename_fn: Callable[[Any], Any],
      adaptation_dataset: tf.data.Dataset,
      modality: Modality,
      multi_label: bool,
      batchwise_metrics: type[clu_metrics.Collection],
      writer: metric_writers.MetricWriter,
      reporter: periodic_actions.ReportProgress,
      use_supervised_metrics: bool,
      **method_kwargs,
  ) -> AdaptationState:
    """Perform an epoch of adaptation.

    Args:
      key: The jax random key used for random operations in this epoch.
      model_bundle: The model_utils.ModelBundle to use for adaptation.
      adaptation_state: The current AdaptationState. Once the epoch is over, an
        update version of it is returned.
      rename_fn: The function to use to rename the parameter keys. This is
        needed if using `optax.multi_transform`, which can be used to define
        different learning rates for different parameters.
      inverse_rename_fn: A callable that reverses the changes of `rename_fn`.
      adaptation_dataset: The dataset used for adaptation.
      modality: The current modality.
      multi_label: Whether the current classification dataset is single-label or
        multi-label. Important to choose the adequate metrics and losses.
      batchwise_metrics: The collection of metrics to keep track of during
        adaptation.
      writer: A MetricWriter that logs all metrics.
      reporter: A ReportProgress that helps keep track of speed of adaptation.
      use_supervised_metrics: Whether the current dataset is supervised or not.
      **method_kwargs: Additional method-specific kwargs.

    Returns:
      An updated version of the adaptation state.

    Raises:
      ValueError: If model_bundle's optimizer is not None and batchwise_metrics
        does not contain a 'main_loss' metric.
    """

    def forward(params, key, batch, model_state, **method_gather_args):
      """Forwards the batch through the current model."""
      dropout_key, low_pass_key = jax.random.split(key)
      variables = {"params": params, **model_state}

      # Foward pass through the model
      if method_kwargs["update_bn_statistics"]:
        model_outputs, model_state = model_bundle.model.apply(
            variables,
            batch[modality.value],
            train=method_kwargs["use_dropout"],
            mutable=list(model_state.keys()),
            use_running_average=False,
            rngs={"dropout": dropout_key, "low_pass": low_pass_key},
        )
      else:
        model_outputs = model_bundle.model.apply(
            variables,
            batch[modality.value],
            use_running_average=True,
            train=method_kwargs["use_dropout"],
            rngs={"dropout": dropout_key, "low_pass": low_pass_key},
        )

      # Compute metrics and loss
      logits2probas = nn.sigmoid if multi_label else nn.softmax
      gather_args = {
          "multi_label": multi_label,
          "outputs": model_outputs,
          "probabilities": logits2probas(model_outputs.label),
          "label_mask": (
              jnp.ones_like(model_outputs.label)
              if "label_mask" not in batch
              else batch["label_mask"]
          ),
      }
      gather_args.update(method_gather_args)
      if use_supervised_metrics:
        gather_args.update({"label": batch["label"].astype(np.int32)})

      # Compute the current metrics
      batch_metrics = batchwise_metrics.gather_from_model_output(
          **gather_args, **method_kwargs
      ).compute()

      # Extract the loss to optimize, and add weight decay.
      if "main_loss" not in batch_metrics:
        if model_bundle.optimizer is not None:
          raise ValueError(
              "Any SFDA method that defines an optimizer should also specify "
              "the key 'main_loss' by overriding 'get_adaptation_metrics'."
          )
        main_loss = None
      else:
        main_loss = batch_metrics["main_loss"]
        if method_kwargs["optimizer_config"].weight_decay > 0.0:
          main_loss += method_kwargs[
              "optimizer_config"
          ].weight_decay * losses.l2_loss(params)
      return main_loss, (batch_metrics, model_state)

    @functools.partial(jax.pmap, axis_name="batch")
    def update_step(
        batch: dict[str, jnp.ndarray],
        adaptation_state: AdaptationState,
        key: jax.random.PRNGKeyArray,
        **method_gather_args,
    ) -> tuple[dict[str, jnp.ndarray], AdaptationState]:
      """Updates the model's state and params using the given batch."""

      params = adaptation_state.model_params
      model_state = adaptation_state.model_state
      opt_state = adaptation_state.opt_state

      if model_bundle.optimizer is not None:
        # If an optimizer is defined, compute gradient transformations.
        # Doing so, get the new model state.
        grads, (batch_metrics, model_state) = jax.grad(forward, has_aux=True)(
            params,
            key=key,
            batch=batch,
            model_state=model_state,
            **method_gather_args,
        )
        grads = jax.lax.pmean(grads, axis_name="batch")

        # Update model's parameters from gradient transformations.
        # Rename params and gradients as the optimizer expects.
        grads = rename_fn(grads, {}, "")
        renamed_params = rename_fn(params, {}, "")
        updates, opt_state = model_bundle.optimizer.update(
            grads, opt_state, renamed_params
        )
        params = optax.apply_updates(renamed_params, updates)
        # Undo the renaming, else the next forward pass will fail.
        params = inverse_rename_fn(params)
      else:
        # Otherwise, we simply forward the data through the model.
        _, (batch_metrics, model_state) = forward(
            params,
            key=key,
            batch=batch,
            model_state=model_state,
            **method_gather_args,
        )

      # Update adaptation state
      adaptation_state = adaptation_state.replace(
          step=adaptation_state.step + 1,
          model_params=params,
          opt_state=opt_state,
          model_state=model_state,
      )
      return batch_metrics, adaptation_state

    # Iterate over batches.
    adaptation_state = flax_utils.replicate(adaptation_state)
    for batch in tqdm.tqdm(
        adaptation_dataset.as_numpy_iterator(), total=len(adaptation_dataset)
    ):
      batch = jax.tree_map(np.asarray, batch)
      verify_batch(batch)
      current_step = int(flax_utils.unreplicate(adaptation_state.step))
      step_key, key = jax.random.split(key)
      step_key = jax.random.split(step_key, num=jax.local_device_count())

      st = time.time()
      adaptation_state, method_gather_args = self.before_iter(
          key=step_key,
          model_bundle=model_bundle,
          adaptation_state=adaptation_state,
          batch=batch,
          modality=modality,
          multi_label=multi_label,
          **method_kwargs,
      )
      elapsed = time.time() - st
      logging.info("sfda_method.before_iter completed in %5.3f", elapsed)

      # Perform the update
      st = time.time()
      update_step = self.cache_get_update_step(update_step)
      batch_metrics, adaptation_state = update_step(
          batch=keep_jax_types(batch),
          adaptation_state=adaptation_state,
          key=step_key,
          **method_gather_args,
      )
      elapsed = time.time() - st
      logging.info("sfda_method update_step completed in %5.3f", elapsed)

      if current_step % 100 == 0:
        st = time.time()
        reporter(current_step)
        writer.write_scalars(
            current_step, flax_utils.unreplicate(batch_metrics)
        )
        elapsed = time.time() - st
        logging.info("sfda_method reporting completed in %5.3f", elapsed)

    return flax_utils.unreplicate(adaptation_state)

  def evaluate(
      self,
      model_bundle: model_utils.ModelBundle,
      writer: jnp.ndarray,
      adaptation_state: AdaptationState,
      eval_dataset: tf.data.Dataset,
      multi_label: bool,
      modality: Modality,
      sample_threshold: int = 5,
      compute_mca: bool = False,
  ) -> None:
    """Evaluate the current adaptation state.

    The writer is in charge of logging all results.

    Args:
      model_bundle: The model_utils.ModelBundle to use for evaluation.
      writer: The evaluation writer.
      adaptation_state: The current AdaptationState to evaluate.
      eval_dataset: The dataset to perform evaluation on.
      multi_label: Whether the problem is multi-label or not. Used to determine
        adequate metrics.
      modality: Which modality are we using.
      sample_threshold: Class that have fewer samples than this thresold are
        discarded when computing cmAP metric in order to reduce noise caused by
        sample size.
      compute_mca: Whether to compute the mean class accuracy metric.
    """

    # Define validation metrics.
    valid_metrics = get_common_metrics(supervised=True, multi_label=multi_label)
    valid_metrics = flax_utils.replicate(valid_metrics.empty())
    cmap_metrics = (
        model_utils.make_cmap_metrics_dict(("label",)) if multi_label else {}
    )
    mca_metric = mca.make_mca_metric() if not multi_label else None

    @functools.partial(jax.pmap, axis_name="batch")
    def update_metrics(
        metric_collection: clu_metrics.Collection,
        batch: dict[str, jnp.ndarray],
        adaptation_state: AdaptationState,
    ):
      variables = {
          "params": adaptation_state.model_params,
          **adaptation_state.model_state,
      }
      model_outputs = model_bundle.model.apply(
          variables,
          batch[modality.value],
          train=False,
          use_running_average=True,
      )
      logits2probas = nn.sigmoid if multi_label else nn.softmax
      return model_outputs, metric_collection.merge(
          metric_collection.gather_from_model_output(
              multi_label=multi_label,
              outputs=model_outputs,
              probabilities=logits2probas(model_outputs.label),
              label_mask=jnp.ones_like(model_outputs.label)
              if "label_mask" not in batch
              else batch["label_mask"],
              label=batch["label"].astype(np.int32),
          )
      )

    update_metrics = self.cache_get_update_metrics(update_metrics)

    current_epoch = int(adaptation_state.epoch)

    # Loop over validation dataset
    for batch in tqdm.tqdm(eval_dataset.as_numpy_iterator()):
      batch = jax.tree_map(np.asarray, batch)
      verify_batch(batch)
      model_outputs, valid_metrics = update_metrics(
          metric_collection=valid_metrics,
          batch=keep_jax_types(batch),
          adaptation_state=flax_utils.replicate(adaptation_state),
      )
      cmap_metrics = model_utils.update_cmap_metrics_dict(
          cmap_metrics, model_outputs, batch
      )
      if compute_mca:
        mca_metric = mca.update_mca_metric(mca_metric, model_outputs, batch)

    # Metrics computations and logging
    valid_metrics = flax_utils.unreplicate(valid_metrics).compute()
    if compute_mca and mca_metric is not None:
      mca_metric_value, per_class_accs_value = mca_metric.compute()
      valid_metrics["mean_class_accuracy"] = mca_metric_value
      for i in range(len(per_class_accs_value)):
        valid_metrics[f"{i}_class_accuracy"] = per_class_accs_value[i]
    valid_metrics = {k.replace("___", "/"): v for k, v in valid_metrics.items()}
    cmap_metrics = flax_utils.unreplicate(cmap_metrics)
    for key in cmap_metrics:
      cmap_value = cmap_metrics[key].compute(sample_threshold=sample_threshold)
      valid_metrics[f"{key}_cmap"] = cmap_value
    if writer is not None:
      writer.write_scalars(current_epoch, valid_metrics)


def perform_adaptation(
    key: jax.random.PRNGKeyArray,
    sfda_method: SFDAMethod,
    adaptation_state: AdaptationState,
    rename_fn: Callable[[Any, Any, str], Any],
    inverse_rename_fn: Callable[[Any], Any],
    adaptation_dataset: tf.data.Dataset,
    use_supervised_metrics: bool,
    validation_dataset: tf.data.Dataset,
    model_bundle: model_utils.ModelBundle,
    logdir: str,
    multi_label: bool,
    modality: Modality,
    eval_every: int,
    eval_mca_every: int,
    **method_kwargs,
) -> AdaptationState:
  """Given the adaptation method and dataset, perform the full adaptation.

  Args:
    key: The initial jax random key to use for random operations.
    sfda_method: The Source-Free Domain Adaptation method to use.
    adaptation_state: The initial AdaptationState to adapt. Once adaptation is
      over, its updated version is returned.
    rename_fn: The function to use to rename the parameter keys. This is needed
      if using `optax.multi_transform`, which can be used to define different
      learning rates for different parameters.
    inverse_rename_fn: A callable that reverses the changes of `rename_fn`.
    adaptation_dataset: The dataset used for adaptation.
    use_supervised_metrics: Whether the current adaptation dataset is supervised
      or not.
    validation_dataset: The dataset used for evaluation.
    model_bundle: The model_utils.ModelBundle to use for adaptation.
    logdir: Where to write logs.
    multi_label: Whether the current problem is multi-label or single-label.
    modality: The current modality used.
    eval_every: Frequency (in epochs) to trigger evaluation.
    eval_mca_every: The frequency (in epochs) to trigger computation of the mean
      class accuracy (mca) metric during evaluation, as long as `eval_every` is
      set to a multiple of this frequency. That is, each time evaluation is
      triggered (based on the value of `eval_every`), the computation of mca may
      also be triggered, if the epoch is also a multiple of `eval_mca_every`.
      Note also that `eval_mca_every` is any negative number, mca will never be
      computed.
    **method_kwargs: Method's additional keywargs.

  Returns:
    An updated version of the Adaptation state.
  """

  # Initialize metrics
  batchwise_metrics = sfda_method.get_adaptation_metrics(
      supervised=use_supervised_metrics,
      multi_label=multi_label,
      **method_kwargs,
  )

  # Logging
  adaptation_writer = metric_writers.create_default_writer(
      logdir, asynchronous=False, collection="adaptation"
  )
  reporter = periodic_actions.ReportProgress(writer=adaptation_writer)

  validation_writer = metric_writers.create_default_writer(
      logdir, asynchronous=False, collection="validation"
  )

  # Before run.
  st = time.time()
  adaptation_state = sfda_method.before_run(
      key=key,
      model_bundle=model_bundle,
      adaptation_state=adaptation_state,
      multi_label=multi_label,
      modality=modality,
      adaptation_dataset=adaptation_dataset,
      **method_kwargs,
  )
  elapsed = time.time() - st
  logging.info("sfda.before_run completed in %5.3f", elapsed)

  for epoch in range(method_kwargs["num_epochs"]):
    # Before every epoch, perform a round of evaluation on the validation set.
    if epoch % eval_every == 0:
      compute_mca = (epoch % eval_mca_every == 0) and eval_mca_every >= 0
      st = time.time()
      sfda_method.evaluate(
          model_bundle=model_bundle,
          adaptation_state=adaptation_state,
          eval_dataset=validation_dataset,
          writer=validation_writer,
          modality=modality,
          multi_label=multi_label,
          compute_mca=compute_mca,
      )
      validation_writer.flush()
      elapsed = time.time() - st
      logging.info("sfda_method.evaluate completed in %5.3f", elapsed)

    st = time.time()
    adaptation_state = sfda_method.before_epoch(
        key=key,
        model_bundle=model_bundle,
        adaptation_state=adaptation_state,
        multi_label=multi_label,
        modality=modality,
        adaptation_dataset=adaptation_dataset,
        writer=adaptation_writer,
        **method_kwargs,
    )
    elapsed = time.time() - st
    logging.info("sfda_method.before_epoch completed in %5.3f", elapsed)

    st = time.time()
    adaptation_state = sfda_method.do_epoch(
        key=key,
        model_bundle=model_bundle,
        adaptation_state=adaptation_state,
        rename_fn=rename_fn,
        inverse_rename_fn=inverse_rename_fn,
        multi_label=multi_label,
        modality=modality,
        adaptation_dataset=adaptation_dataset,
        batchwise_metrics=batchwise_metrics,
        writer=adaptation_writer,
        reporter=reporter,
        workdir=logdir,
        use_supervised_metrics=use_supervised_metrics,
        **method_kwargs,
    )
    elapsed = time.time() - st
    logging.info("sfda_method.do_epoch completed in %5.3f", elapsed)
    adaptation_state = adaptation_state.replace(
        epoch=adaptation_state.epoch + 1
    )
    adaptation_writer.flush()

  # When adaptation is finished, we perform a final round of evaluation on the
  # validation set.
  sfda_method.evaluate(
      model_bundle=model_bundle,
      adaptation_state=adaptation_state,
      eval_dataset=validation_dataset,
      writer=validation_writer,
      modality=modality,
      multi_label=multi_label,
      compute_mca=eval_mca_every >= 0,
  )

  adaptation_writer.close()
  validation_writer.close()
  return adaptation_state


def keyed_map(
    key: str, outputs: output.AnyOutput, **kwargs
) -> jnp.ndarray | None:
  label_mask = kwargs.get(key + "_mask", None)
  return chirp_metrics.average_precision(
      scores=getattr(outputs, key), labels=kwargs[key], label_mask=label_mask
  )


def get_common_metrics(
    supervised: bool, multi_label: bool
) -> type[clu_metrics.Collection]:
  """Obtain a common set of metrics and losses.

  Args:
    supervised: Whether the dataset over which those metrics will be tracked has
      labels or not.
    multi_label: Whether the current problem is multi-label or single-label.

  Returns:
    A collection of metrics.
  """
  metrics_dict = {}
  if supervised:
    if multi_label:
      metrics_dict["label_map"] = clu_metrics.Average.from_fun(
          functools.partial(keyed_map, key="label")
      )
      metrics_dict["supervised_loss"] = clu_metrics.Average.from_fun(
          losses.label_binary_xent
      )
      metrics_dict["entropy_loss"] = clu_metrics.Average.from_fun(
          losses.label_binary_ent
      )
      metrics_dict["marginal_entropy"] = metrics.MarginalBinaryEntropy
    else:
      metrics_dict["supervised_loss"] = clu_metrics.Average.from_fun(
          losses.label_xent
      )
      metrics_dict["entropy_loss"] = clu_metrics.Average.from_fun(
          losses.label_ent
      )
      metrics_dict["accuracy"] = metrics.Accuracy
      metrics_dict["marginal_entropy"] = metrics.MarginalEntropy

  return clu_metrics.Collection.create(**metrics_dict)


def verify_batch(batch: dict[str, jnp.ndarray]) -> None:
  """Performs non-jittable verifications on a batch of data.

  Args:
    batch: The current batch of data.

  Raises:
    ValueError: If the batch has a label_mask, and label_mask differs across
      samples.
  """
  if "label_mask" in batch:
    label_mask = flax_utils.unreplicate(batch["label_mask"])
    if not (
        jnp.tile(label_mask[0], (label_mask.shape[0], 1)) == label_mask
    ).all():
      raise ValueError(
          "Some metrics (e.g. marginal entropy) can only be computed if each "
          "sample's probability distribution is defined over the same set of"
          "classes. Therefore, we verify that `label_mask` is the same across "
          "samples."
      )


def batch_forward(
    model: nn.Module,
    modality: Modality,
    use_batch_statistics: bool,
    train: bool = False,
) -> ForwardStepType:
  """Collects the model's output on the current batch of data.

  Args:
    model: The model.
    modality: The modality used.
    use_batch_statistics: Whether to use BatchNorm's running statistics, or the
      batch's statistics.
    train: Whether to use the model in training mode. Default to False, as this
      function is nominally used to compute pseudo-labels (e.g. Teacher step of
      Notela), which usually removes any source of noise (including dropout).

  Returns:
    Batch forward step callable.
  """

  @jax.pmap
  def forward(batch, model_state, params, rngs):
    if use_batch_statistics:
      outputs, _ = model.apply(
          {"params": params, **model_state},
          batch[modality.value],
          train=train,
          mutable=list(model_state.keys()),
          use_running_average=False,
          rngs=rngs,
      )
    else:
      outputs = model.apply(
          {"params": params, **model_state},
          batch[modality.value],
          use_running_average=True,
          train=train,
          rngs=rngs,
      )
    return outputs

  return forward
