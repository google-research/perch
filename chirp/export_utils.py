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

"""Common utilities for exporting SavedModels and TFLite models."""

import os
from typing import Sequence

from absl import logging
from chirp.taxonomy import namespace
from jax.experimental import jax2tf
import tensorflow as tf


class Jax2TfModelWrapper(tf.Module):
  """Wrapper for Jax models for exporting with variable input shape."""

  def __init__(
      self,
      infer_fn,
      jax_params,
      input_shape: Sequence[int | None],
      enable_xla: bool = False,
      coord_ids: str = 'bt',
      name=None,
  ):
    """Initialize the wrapper.

    Args:
      infer_fn: The inference function for the Jax model.
      jax_params: Parameters (ie, model weights) for the Jax model.
      input_shape: Input shape, with 'None' for any axes which will be variable.
      enable_xla: Whether to use XLA ops in the exported model. Defaults to
        False, which is necessary for subsequent TFLite conversion.
      coord_ids: String with length matching the length of the input_shape, used
        for identifying polymorphic shape parameters.
      name: Model name.
    """
    super(Jax2TfModelWrapper, self).__init__(name=name)
    # The automatically generated variable names in the checkpoint end up being
    # very uninformative. There may be a good way to map in better names.
    self._structured_variables = tf.nest.map_structure(tf.Variable, jax_params)
    self.input_shape = input_shape

    # Construct the jax polymorphic shape.
    jp_shape = []
    for i, s in enumerate(input_shape):
      if s is None:
        jp_shape.append(coord_ids[i])
      else:
        jp_shape.append('_')
    jp_shape = '(' + ','.join(jp_shape) + ')'

    # The variables structure needs to be flattened for the saved_model.
    self._variables = tf.nest.flatten(self._structured_variables)
    logging.info('Running jax2tf conversion...')
    converted_infer_fn = jax2tf.convert(
        infer_fn,
        enable_xla=enable_xla,
        with_gradient=False,
        polymorphic_shapes=[jp_shape, None],
    )
    infer_partial = lambda inputs: converted_infer_fn(  # pylint:disable=g-long-lambda
        inputs, self._structured_variables
    )
    self.infer_tf = tf.function(
        infer_partial,
        jit_compile=True,
        input_signature=[tf.TensorSpec(input_shape)],
    )
    logging.info('Jax2TfModelWrapper initialized.')

  def __call__(self, inputs):
    return self.infer_tf(inputs)

  def get_tf_zero_inputs(self):
    """Construct some dummy inputs with self.input_shape."""
    fake_shape = []
    for s in self.input_shape:
      if s is None:
        fake_shape.append(1)
      else:
        fake_shape.append(s)
    return tf.zeros(fake_shape)

  def export_converted_model(
      self,
      export_dir: str,
      train_step: int,
      class_lists: dict[str, namespace.ClassList] | None = None,
      export_tf_lite: bool = True,
      tf_lite_dtype: str = 'float16',
      tf_lite_select_ops: bool = True,
  ):
    """Export converted TF models."""
    fake_inputs = self.get_tf_zero_inputs()
    logging.info('Creating concrete function...')
    concrete_fn = self.infer_tf.get_concrete_function(fake_inputs)

    logging.info('Saving TF SavedModel...')
    tf.saved_model.save(
        self, os.path.join(export_dir, 'savedmodel'), signatures=concrete_fn
    )
    with tf.io.gfile.GFile(
        os.path.join(export_dir, 'savedmodel', 'ckpt.txt'), 'w'
    ) as f:
      f.write(f'train_state.step: {train_step}\n')

    logging.info('Writing class lists...')
    if class_lists is not None:
      for key, class_list in class_lists.items():
        with tf.io.gfile.GFile(
            os.path.join(export_dir, f'{key}.csv'), 'w'
        ) as f:
          # NOTE: Although the namespace is written to the file, there is no
          # guarantee that the class list will still be compatible with the
          # namespace if the latter gets updated.
          f.write(class_list.to_csv())

    if not export_tf_lite:
      logging.info('Skipping TFLite export.')
      logging.info('Export complete.')
      return
    # Export TFLite model.
    logging.info('Converting to TFLite...')
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_fn], self
    )

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if tf_lite_dtype == 'float16':
      converter.target_spec.supported_types = [tf.float16]
    elif tf_lite_dtype == 'float32':
      converter.target_spec.supported_types = [tf.float32]
    elif tf_lite_dtype == 'auto':
      # Note that the default with optimizations is int8, which requires further
      # tuning.
      pass
    else:
      raise ValueError(f'Unsupported dtype: {tf_lite_dtype}')
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    ]
    if tf_lite_select_ops:
      converter.target_spec.supported_ops += [tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_float_model = converter.convert()

    if not tf.io.gfile.exists(export_dir):
      tf.io.gfile.makedirs(export_dir)
    with tf.io.gfile.GFile(os.path.join(export_dir, 'model.tflite'), 'wb') as f:
      f.write(tflite_float_model)
    with tf.io.gfile.GFile(
        os.path.join(export_dir, 'tflite_ckpt.txt'), 'w'
    ) as f:
      f.write(f'train_state.step: {train_step}\n')

    logging.info('Export complete.')
