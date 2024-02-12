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

"""Model utils and colab helpers."""

import collections
import glob
import json
import os

from absl import logging
from chirp.birb_sep_paper import audio_ops
from etils import epath
from ml_collections import config_dict
import numpy as np
import tensorflow

tf = tensorflow.compat.v1
tf2 = tensorflow.compat.v2

ClassifierState = collections.namedtuple(
    'ClassifierState',
    [
        'session',
        'audio_placeholder',
        'melspec_placeholder',
        'hints_placeholder',
        'melspec_output',
        'logits',
        'all_outputs',
    ],
)

SeparatorState = collections.namedtuple(
    'SeparatorState', ['session', 'audio_placeholder', 'output_tensor']
)


def load_params_from_json(model_dir, filename='hyper_params.json'):
  """Read hyperparams from json file in the model_dir."""

  if os.path.exists(os.path.join(model_dir, filename)):
    filepath = os.path.join(model_dir, filename)
  elif os.path.exists(os.path.join(model_dir, 'run_00', filename)):
    filepath = os.path.join(model_dir, 'run_00', filename)
  else:
    raise ValueError('Could not find hyper_params file.')
  with tf2.io.gfile.GFile(filepath) as f:
    json_str = f.read()
  params_dict = json.loads(json_str)
  return config_dict.ConfigDict(params_dict)


def audio_to_input_fn(
    audio,
    dataset_params,
    interval_s=4,
    sample_rate_hz=44100,
    max_intervals=10,
    batch_size=None,
    hints=None,
):
  """Perform peak-finding segmentation, batch segments."""
  if batch_size is None:
    batch_size = 4
  intervals = audio_ops.SlicePeakedAudio(
      audio,
      sample_rate_hz=sample_rate_hz,
      interval_s=interval_s,
      max_intervals=max_intervals,
  )
  audio_batch = np.concatenate(
      [np.expand_dims(v, 0) for v in intervals.values()], axis=0
  )
  if hints is None:
    hints = np.ones([batch_size, dataset_params.n_classes])

  def _map_features(features):
    ms = audio_ops.GetAugmentedMelspec(
        features['audio'],
        dataset_params.sample_rate_hz,
        dataset_params.melspec_params,
        dataset_params.feature_cleaning,
        dataset_params.filter_augment,
    )
    return {
        'audio': features['audio'],
        'melspec': ms,
        'hints': hints,
    }

  def input_fn(params):
    """Input function wrapping the intervals."""
    params = config_dict.ConfigDict(params)
    dataset = tf.data.Dataset.from_tensors({'audio': np.float32(audio_batch)})
    dataset = dataset.map(_map_features)
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size)
    return dataset

  return input_fn, intervals.keys()


def build_optimizer(learning_rate, use_tpu):
  """build the optimizer."""
  print('Defining optimizer...')
  with tf.variable_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=0.01
    )
    # Importing contrib_estimator now fails.
    # if clip_gradient > 0:
    #   optimizer = contrib_estimator.clip_gradients_by_norm(
    #       optimizer, clip_gradient)
    if use_tpu:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)
  return optimizer


def mean_reciprocal_rank(logits, labels):
  asortd = tf.argsort(logits, axis=1)
  asortd = tf.argsort(asortd, axis=1)
  asortd = -(asortd - logits.shape[1])
  rnks = tf.reduce_sum(tf.to_float(asortd) * tf.to_float(labels), axis=1)
  invrnks = tf.reciprocal(rnks)
  return tf.reduce_mean(invrnks)


def map_k(labels_onehot, logits, k=1, name=''):
  """Finds mean average precision at k."""
  # Need to convert one_hot labels to class ids.
  labels_onehot = tf.cast(labels_onehot, tf.int64)
  class_ids = tf.expand_dims(
      tf.range(labels_onehot.shape[-1], dtype=tf.int64), 0
  )
  masked_class_ids = labels_onehot * class_ids
  # Set the false labels to -1, since the zero label is allowed.
  masked_class_ids += (labels_onehot - 1) * tf.ones(
      labels_onehot.shape, tf.int64
  )
  final_map_k, map_k_update_op = tf.metrics.average_precision_at_k(
      masked_class_ids, logits, k, name=name
  )
  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, map_k_update_op)
  return final_map_k


def _find_checkpoint(model_path: str) -> str:
  # Publicly released model does not have a checkpoints directory file.
  ckpt = None
  for ckpt in sorted(tuple(epath.Path(model_path).glob('model.ckpt-*.index'))):
    ckpt = ckpt.as_posix()[: -len('.index')]
  if ckpt is None:
    raise FileNotFoundError('Could not find checkpoint file.')
  return ckpt


def load_separation_model(model_path):
  """Loads a separation model graph for inference."""
  metagraph_path_ns = os.path.join(model_path, 'inference.meta')
  checkpoint_path = _find_checkpoint(model_path)
  graph_ns = tf.Graph()
  sess_ns = tf.compat.v1.Session(graph=graph_ns)
  with graph_ns.as_default():
    new_saver = tf.train.import_meta_graph(metagraph_path_ns)
    new_saver.restore(sess_ns, checkpoint_path)
    input_placeholder_ns = graph_ns.get_tensor_by_name(
        'input_audio/receiver_audio:0'
    )
    output_tensor_ns = graph_ns.get_tensor_by_name('denoised_waveforms:0')
  return SeparatorState(sess_ns, input_placeholder_ns, output_tensor_ns)


def load_saved_model(sess, model_dir, inference_subdir='inference'):
  """Loads a model from a saved_model.pb file.

  Args:
    sess: The TensorFlow session where the loaded model will be run.
    model_dir: Model directory.
    inference_subdir: Subdirectory containing the saved_model.pb file.

  Returns:
    signature_def: A ProtoBuf of the signature definition from the
      loaded graph definition.
  """

  load_dir = os.path.join(model_dir, inference_subdir)

  meta_graph_def = tf.saved_model.load(sess, [tf.saved_model.SERVING], load_dir)
  signature_def = meta_graph_def.signature_def[
      tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  ]
  return signature_def


def load_classifier_state(model_path, sample_rate=22050):
  """Load all classifier state for the given xid + run_num."""
  cl_graph = tf.Graph()
  cl_sess = tf.Session(graph=cl_graph)
  with cl_graph.as_default():
    # Need to convert to melspec for the classifier.
    audio_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None])
    # TODO(tmd): Should read these from the model params instead of hardcoding.
    melspec_params = {
        'melspec_frequency': 100,
        'upper_edge_hertz': 10000.0,
        'scaling': 'pcen',
    }
    feature_cleaning = {
        'strategy': 'whiten',
        'clean_thresh': 1.0,
    }
    cl_melspec = audio_ops.GetAugmentedMelspec(
        audio_placeholder, sample_rate, melspec_params, feature_cleaning, None
    )

    signature_def = load_saved_model(cl_sess, model_path)
    saved_model_melspec_input = signature_def.inputs['melspec_input'].name
    output_logits = signature_def.outputs['logits_label_prediction'].name
    if 'hints_input' in signature_def.inputs:
      hints_input = signature_def.inputs['hints_input'].name
    else:
      hints_input = None
    all_outputs = {
        k: signature_def.outputs[k].name for k in signature_def.outputs
    }
  return ClassifierState(
      cl_sess,
      audio_placeholder,
      saved_model_melspec_input,
      hints_input,
      cl_melspec,
      output_logits,
      all_outputs,
  )


def load_classifier_ensemble(model_path, sample_rate=22050, max_runs=5):
  """Loads ensemble of classifiers."""
  classifiers = {}
  runs = glob.glob(os.path.join(model_path, 'run_*'))
  runs = [r for r in runs if '.xid' not in r]
  if not runs:
    logging.info('Loading single classifier : %s', model_path)
    classifiers['run_00'] = load_classifier_state(model_path, sample_rate)
  else:
    for run in runs[:max_runs]:
      run_path = os.path.join(run)
      logging.info('loading classifier : %s', run_path)
      classifiers[run] = load_classifier_state(run_path, sample_rate)
  return classifiers


def ensemble_classify(
    audio_batch, classifier_states, hints=None, logits_key=None
):
  """Classify a batch of audio with the given set of classifiers."""
  all_logits = None
  if hints is not None and len(hints.shape) == 1:
    # tile the hints.
    hints = hints[np.newaxis, :]
    hints = np.tile(hints, [audio_batch.shape[0], 1])

  cl0 = list(classifier_states.values())[0]
  melspec = cl0.session.run(
      cl0.melspec_output, feed_dict={cl0.audio_placeholder: audio_batch}
  )

  for cl_state in classifier_states.values():
    if logits_key is None:
      target = cl_state.logits
    else:
      target = cl_state.all_outputs[logits_key]
    if cl_state.hints_placeholder is not None:
      got_logits = cl_state.session.run(
          target,
          feed_dict={
              cl_state.melspec_placeholder: melspec,
              cl_state.hints_placeholder: hints,
          },
      )
    else:
      got_logits = cl_state.session.run(
          target, feed_dict={cl_state.melspec_placeholder: melspec}
      )

    got_logits = got_logits[:, np.newaxis]
    if all_logits is None:
      all_logits = got_logits
    else:
      all_logits = np.concatenate([all_logits, got_logits], axis=1)
  return melspec, all_logits


def model_embed(
    audio_batch, classifier_state, hints=None, output_key='pooled_embedding'
):
  """Use ClassifierState to compute an audio embedding."""
  if hints is not None and len(hints.shape) == 1:
    # tile the hints.
    hints = hints[np.newaxis, :]
    hints = np.tile(hints, [audio_batch.shape[0], 1])
  melspec = classifier_state.session.run(
      classifier_state.melspec_output,
      feed_dict={classifier_state.audio_placeholder: audio_batch},
  )
  if classifier_state.hints_placeholder is not None:
    embedding = classifier_state.session.run(
        classifier_state.all_outputs[output_key],
        feed_dict={
            classifier_state.melspec_placeholder: melspec,
            classifier_state.hints_placeholder: hints,
        },
    )
  else:
    embedding = classifier_state.session.run(
        classifier_state.logits,
        feed_dict={classifier_state.melspec_placeholder: melspec},
    )
  return embedding


def progress_dot(i, verbose=True, div=1):
  """Print a dot every so often, in rows of one hundred."""
  if not verbose:
    return
  if div > 1:
    i = i // div
  elif (i + 1) % 1000 == 0:
    print('*')
  elif (i + 1) % 50 == 0:
    print('.')
  elif (i + 1) % 25 == 0:
    print('.', end=' ')
  else:
    print('.', end='')


def ensemble_classify_batched(
    audios, classifier_states, hints=None, batch_size=32, verbose=True
):
  """Ensemble classify by batching input audio."""
  logits = None
  mels = None
  ds = tf.data.Dataset.from_tensor_slices(audios).batch(batch_size)
  for i, batch in enumerate(ds):
    new_mels, new_logits = ensemble_classify(
        batch.numpy(), classifier_states, hints
    )
    if logits is None:
      logits = new_logits
      mels = new_mels
    else:
      logits = np.concatenate([logits, new_logits], axis=0)
      mels = np.concatenate([mels, new_mels], axis=0)
    progress_dot(i, verbose)
  return mels, logits


def separate_windowed(
    audio, separator_state, hop_size_s=2.5, window_size_s=5, sample_rate=22050
):
  """Separate a large audio file in windowed chunks."""
  start_sample = 0
  window_size = int(window_size_s * sample_rate)
  hop_size = int(hop_size_s * sample_rate)

  # Separate audio.
  sep_chunks = []
  raw_chunks = []
  while start_sample + window_size <= audio.shape[0] or not raw_chunks:
    audio_chunk = audio[start_sample : start_sample + window_size]
    raw_chunks.append(audio_chunk[np.newaxis, :])
    separated_audio = separator_state.session.run(
        separator_state.output_tensor,
        feed_dict={
            separator_state.audio_placeholder: audio_chunk[
                np.newaxis, np.newaxis, :
            ]
        },
    )
    sep_chunks.append(separated_audio)
    start_sample += hop_size
  if not raw_chunks:
    return None, None
  raw_chunks = np.concatenate(raw_chunks, axis=0)
  sep_chunks = np.concatenate(sep_chunks, axis=0)
  return sep_chunks, raw_chunks


def separate_classify(
    audio,
    classifier_states,
    separator_state,
    hints=None,
    batch_size=4,
    hop_size_s=2.5,
    window_size_s=5,
    sample_rate=22050,
    verbose=False,
):
  """Separate and classify an audio array."""
  sep_chunks, raw_chunks = separate_windowed(
      audio, separator_state, hop_size_s, window_size_s, sample_rate
  )
  if raw_chunks is None:
    return None, None

  # Run classifiers on chunks.
  big_batch = np.reshape(
      sep_chunks, [sep_chunks.shape[0] * sep_chunks.shape[1], -1]
  )
  sep_mels, sep_logits = ensemble_classify_batched(
      big_batch,
      classifier_states,
      hints=hints,
      batch_size=batch_size,
      verbose=verbose,
  )
  sep_mels = np.reshape(
      sep_mels,
      [
          sep_chunks.shape[0],
          sep_chunks.shape[1],
          sep_mels.shape[-2],
          sep_mels.shape[-1],
      ],
  )
  sep_logits = np.reshape(
      sep_logits,
      [
          sep_chunks.shape[0],
          sep_chunks.shape[1],
          len(classifier_states),
          sep_logits.shape[-1],
      ],
  )
  raw_mels, raw_logits = ensemble_classify_batched(
      raw_chunks,
      classifier_states,
      hints=hints,
      batch_size=batch_size,
      verbose=verbose,
  )
  raw_logits = raw_logits[:, np.newaxis, :]

  stacked_mels = np.concatenate([raw_mels[:, np.newaxis], sep_mels], axis=1)
  stacked_logits = np.concatenate([raw_logits, sep_logits], axis=1)
  reduced_logits = np.mean(stacked_logits, axis=2)
  reduced_logits = np.max(reduced_logits, axis=1)
  # Use the raw_logits score for the unknown class.
  reduced_logits[:, 0] = np.mean(raw_logits, axis=1)[:, 0, 0]
  return stacked_mels, reduced_logits


def saved_model_prediction(model_dir, mels, hints=None, batch_size=8):
  """Run inference on the set of numpy melspec features."""
  scores = []
  with tf.Graph().as_default():
    with tf.Session() as sess:
      signature_def = load_saved_model(sess, model_dir)
      ms_name = signature_def.inputs['melspec_input'].name
      output_name = signature_def.outputs['logits_label_prediction'].name
      use_hints = 'hints_input' in signature_def.inputs

      # Handle species hinting inputs.
      if use_hints:
        n_classes = (
            signature_def.inputs['hints_input'].tensor_shape.dim[-1].size
        )
        hints = np.ones([n_classes], np.float32)

      dataset_dict = {
          'melspec': mels,
      }
      dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)
      dataset = dataset.batch(batch_size)
      it = dataset.make_one_shot_iterator().get_next()
      while True:
        try:
          features = sess.run(it)
        except tf.errors.OutOfRangeError:
          break
        feed = {ms_name: features['melspec']}
        if use_hints:
          hint_name = signature_def.inputs['hints_input'].name
          batch_hints = np.tile(
              hints[np.newaxis, :], [features['melspec'].shape[0], 1]
          )
          feed[hint_name] = batch_hints

        pred = sess.run(output_name, feed_dict=feed)
        scores.append(pred)
  return np.concatenate(scores, axis=0)


def add_histogram(name, tensor, use_tpu, histogram_vars, add_tpu_summary=False):
  tensor = tf.check_numerics(tensor, 'check ' + name)
  if not use_tpu:
    tf.summary.histogram(name, tensor)
  elif add_tpu_summary:
    histogram_vars.append((name, tensor))
  return tensor


def add_scalar(name, tensor, scalar_vars, use_tpu):
  if use_tpu:
    scalar_vars.append((name, tf.expand_dims(tensor, 0)))
  else:
    tf.summary.scalar(name, tensor)
  return tensor


def make_eval_metrics(mode_key, model_dir, eval_dict):
  """Create an eval metrics map."""
  tensor_map = {k: tf.expand_dims(v, 0) for k, v in eval_dict.items()}
  tensor_map['global_step'] = tf.expand_dims(
      tf.train.get_or_create_global_step(), 0
  )
  summary_path = os.path.join(model_dir, mode_key)
  tf.logging.info('eval_metrics summary path: %s', summary_path)

  def eval_metrics_fn(**tensor_map):
    """Eval function for CPU summaries."""
    tf.logging.info('eval_metrics tensors: %s', tensor_map)
    writer = tf2.summary.create_file_writer(summary_path, max_queue=1000)
    eval_metric_ops = {}

    with writer.as_default():
      for name, tensor in tensor_map.items():
        if name == 'global_step':
          continue
        eval_metric_ops[name] = tf.metrics.mean(tensor)
    return eval_metric_ops

  return eval_metrics_fn, tensor_map
