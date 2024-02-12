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

r"""Inference.

Example command line:
python -m beam_inference \
--source_files=test_files/*.wav \
--model_path=models/taxo_lori_34078450 \
--separation_model_path=models/separator4 \
--output_dir=lorikeet_inference \
--target_species=blakit1 \
--hints_tag=eaus \
--min_logit=-3.0
"""

import collections
import gc
import glob
import os
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import data_tools
import model_utils
import numpy as np
import taxonomy

FLAGS = flags.FLAGS
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SPECIES_INFO_PATH = os.path.join(DATA_PATH, 'species_info.csv')

ENSEMBLE_SIZE = 3

flags.DEFINE_list('source_files', [], 'Source audio files (wav or mp3).')
flags.DEFINE_string(
    'model_path', '', 'Where to find the model params and inference.pb'
)
flags.DEFINE_string(
    'separation_model_path', '', 'Where to find the separation inference model.'
)
flags.DEFINE_string(
    'target_species', '', 'Species code for single-species mode.'
)
flags.DEFINE_string('output_dir', '', 'Where to dump output data.')
flags.DEFINE_integer('num_shards', 2000, 'Number of CSV output shards.')
flags.DEFINE_float(
    'min_logit',
    -1.0,
    'Only emit predictions if a logit is above this threshold.',
)
flags.DEFINE_integer(
    'file_shards', 48, 'Number of sub-jobs to divide each input file into.'
)
flags.DEFINE_string('hints_tag', '', 'Species set tag for hints.')
flags.DEFINE_boolean('dry_run', False, 'Whether to exit after dry-run.')

PredictionTuple = collections.namedtuple(
    'PredictionTuple', ['file_id', 'start_time', 'end_time', 'logits']
)


class InferenceFn(beam.DoFn):
  """Beam function for model inference."""

  def __init__(
      self,
      model_path,
      separation_model_path=None,
      min_logit=-20.0,
      target_species='',
      sample_rate=22050,
      hints_tag=None,
  ):
    # Get a local copy of the inference.pb file.
    self.model_path = model_path
    self.separation_model_path = separation_model_path
    self.inference_path = os.path.join(model_path, 'inference')
    self.min_logit = min_logit
    self.target_species = target_species
    self.sample_rate = sample_rate
    self.hints_tag = hints_tag

  def setup(self):
    # tf.compat.v1.disable_eager_execution()
    # admittedly a bit brittle...
    self.model_params = model_utils.load_params_from_json(self.model_path)
    self.taxo = taxonomy.Taxonomy(self.model_path, DATA_PATH, SPECIES_INFO_PATH)
    if self.target_species and self.target_species not in self.taxo.label_enum:
      raise ValueError(
          'Target species %s not found in taxonomy label enum.'
          % self.target_species
      )
    self.hints = self.taxo.MakeSpeciesHints(species_list_tag=self.hints_tag)

    if self.separation_model_path:
      self.separation_model = model_utils.load_separation_model(
          self.separation_model_path
      )
    self.classifiers = model_utils.load_classifier_ensemble(
        self.model_path, max_runs=ENSEMBLE_SIZE
    )

  def get_hints(self, batch_size):
    if self.hints is not None:
      hints = self.hints[np.newaxis, :]
      hints = np.tile(hints, [batch_size, 1])
    else:
      hints = np.ones([batch_size, self.taxo.NumLabels()])
    return hints

  def infer_target_species(self, file_id, audio, timestamp_offset):
    """Create full taxonomy logits for the target species."""
    window_size_s = self.model_params.dataset.window_size_s
    hop_size_s = window_size_s / 2
    logging.info('...starting separation (%s)', file_id)
    sep_chunks, raw_chunks = model_utils.separate_windowed(
        audio,
        self.separation_model,
        hop_size_s,
        window_size_s,
        self.sample_rate,
    )
    raw_chunks = raw_chunks[:, np.newaxis, :]
    stacked_chunks = np.concatenate([raw_chunks, sep_chunks], axis=1)
    n_chunks = stacked_chunks.shape[0]
    n_channels = stacked_chunks.shape[1]
    big_batch = np.reshape(stacked_chunks, [n_chunks * n_channels, -1])

    # We often get memory blowups at this point; trigger a garbage collection.
    gc.collect()

    sp_info = self.taxo.species_info[self.target_species]
    indices = {
        'label': self.taxo.label_enum[self.target_species],
        'genus': self.taxo.genus_enum[sp_info['genus']],
        'family': self.taxo.family_enum[sp_info['family']],
        'order': self.taxo.order_enum[sp_info['order']],
    }
    target_taxo_logits = {}
    for logits_key, key_index in indices.items():
      logging.info('...starting classification (%s, %s)', file_id, logits_key)
      _, logits = model_utils.ensemble_classify(
          big_batch,
          self.classifiers,
          hints=self.get_hints(big_batch.shape[0]),
          logits_key=logits_key,
      )
      unbatched_logits = np.reshape(
          logits,
          [
              n_chunks,
              n_channels,
              logits.shape[1],  # ensemble
              logits.shape[2],  # num classes
          ],
      )
      # Take the mean logits over the ensemble.
      unbatched_logits = np.mean(unbatched_logits, axis=2)
      # Take the max logit over all separated and raw channels.
      unbatched_logits = np.max(unbatched_logits, axis=1)
      # Choose the logits for the target species.
      target_logits = unbatched_logits[:, key_index]
      # Apply time averaging.
      target_logits = (target_logits[:-1] + target_logits[1:]) / 2
      target_taxo_logits[logits_key] = target_logits

    # All taxo logits should have the same shape: [T]
    # Assemble into a single array.
    all_logits = [
        target_taxo_logits['label'][:, np.newaxis],
        target_taxo_logits['genus'][:, np.newaxis],
        target_taxo_logits['family'][:, np.newaxis],
        target_taxo_logits['order'][:, np.newaxis],
    ]
    all_logits = np.concatenate(all_logits, axis=1)

    for i in range(all_logits.shape[0]):
      if np.max(all_logits[i]) < self.min_logit:
        continue
      beam.metrics.Metrics.counter('beaminference', 'predictions').inc()
      time_stamp = (i + 1) * hop_size_s + (timestamp_offset / self.sample_rate)
      prediction = PredictionTuple(
          file_id, time_stamp, time_stamp + hop_size_s, all_logits[i]
      )
      yield prediction_to_csv(prediction)
    beam.metrics.Metrics.counter('beaminference', 'files_processed').inc()

  def infer_all(
      self,
      audio_filepath,
      audio,
      file_id,
      window_size_s,
      hop_size_s,
      timestamp_offset,
  ):
    """Create label logits for all species."""
    start = time.time()
    logging.info('...starting separate+classify (%s)', file_id)
    _, reduced_logits = model_utils.separate_classify(
        audio,
        self.classifiers,
        self.separation_model,
        hop_size_s=hop_size_s,
        window_size_s=window_size_s,
        sample_rate=self.sample_rate,
        hints=self.get_hints(1),
    )
    elapsed = time.time() - start
    logging.info('finished separate+classify. %5.3fs elsapsed', elapsed)
    beam.metrics.Metrics.distribution(
        'beaminference', 'inference_duration_s'
    ).update(elapsed)
    if reduced_logits is None:
      beam.metrics.Metrics.counter('beaminference', 'no_logits_returned').inc()
      logging.error('no logits from inference : %s', audio_filepath)
      return
    time_averaged_logits = (reduced_logits[:-1] + reduced_logits[1:]) / 2
    for i in range(time_averaged_logits.shape[0]):
      if np.max(time_averaged_logits[i]) < self.min_logit:
        continue
      beam.metrics.Metrics.counter('beaminference', 'predictions').inc()
      time_stamp = (i + 1) * hop_size_s + (timestamp_offset / self.sample_rate)
      prediction = PredictionTuple(
          file_id, time_stamp, time_stamp + hop_size_s, time_averaged_logits[i]
      )
      yield prediction_to_csv(prediction)
    beam.metrics.Metrics.counter('beaminference', 'files_processed').inc()

  @beam.typehints.with_output_types(Any)
  def process(self, source_info, crop_s=-1):
    audio_filepath, shard_num, num_shards = source_info
    file_name = os.path.basename(audio_filepath)
    file_id = file_name.split('.')[0]
    # self.sample_rate = self.model_params.dataset.sample_frequency
    window_size_s = self.model_params.dataset.window_size_s
    hop_size_s = window_size_s / 2

    try:
      logging.info('...loading audio (%s)', audio_filepath)
      audio, _ = data_tools.LoadAudio(audio_filepath, self.sample_rate)
    except Exception as e:  # pylint: disable=broad-except
      beam.metrics.Metrics.counter('beaminference', 'load_audio_error').inc()
      logging.error('Failed to load audio : %s', audio_filepath)
      logging.exception('Load audio exception : %s', e)
      return

    if audio.shape[0] < 2 * self.model_params.dataset.window_size:
      beam.metrics.Metrics.counter('beaminference', 'short_audio_error').inc()
      logging.error('short audio file : %s', audio_filepath)
      return

    if num_shards > 1:
      shard_len = audio.shape[0] // num_shards
      timestamp_offset = shard_num * shard_len
      audio = audio[timestamp_offset : timestamp_offset + shard_len]
    else:
      timestamp_offset = 0

    if crop_s > 0:
      audio = audio[: crop_s * self.sample_rate]

    if self.target_species:
      for pred in self.infer_target_species(file_id, audio, timestamp_offset):
        yield pred
    else:
      for pred in self.infer_all(
          audio_filepath,
          audio,
          file_id,
          window_size_s,
          hop_size_s,
          timestamp_offset,
      ):
        yield pred


def prediction_to_csv(prediction):
  logits = ['%1.3f' % l for l in prediction.logits]
  csv_row = ','.join(
      [
          prediction.file_id,
          '%1.5f' % prediction.start_time,
          '%1.5f' % prediction.end_time,
      ]
      + logits
  )
  return csv_row


def get_counter(metrics, name):
  counter = metrics.query(beam.metrics.MetricsFilter().with_name(name))[
      'counters'
  ]
  if not counter:
    return 0
  return counter[0].result


def main(unused_argv):
  source_files = []
  for pattern in FLAGS.source_files:
    source_files += glob.glob(pattern)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  print('Found %d source files.' % len(source_files))

  source_file_splits = []
  for s in source_files:
    for i in range(FLAGS.file_shards):
      source_file_splits.append((s, i, FLAGS.file_shards))

  # Dry-run.
  print('Starting dry run...')
  test_fn = InferenceFn(
      FLAGS.model_path,
      FLAGS.separation_model_path,
      target_species=FLAGS.target_species,
      hints_tag=FLAGS.hints_tag,
  )
  test_fn.setup()
  got_results = False
  start = time.time()
  print(source_file_splits[15])
  for p in test_fn.process(source_file_splits[15], crop_s=10):
    got_results = True
    print(p)
  elapsed = time.time() - start
  if not got_results:
    raise Exception('Something went wrong; no results found.')
  test_fn.teardown()
  print('Dry run successful! Party! Inference time : %5.3f' % elapsed)
  if FLAGS.dry_run:
    return

  output_prefix = os.path.join(FLAGS.output_dir, 'predictions')
  pipeline = beam.Pipeline()
  _ = (
      pipeline
      | beam.Create(source_file_splits)
      | beam.ParDo(
          InferenceFn(
              FLAGS.model_path,
              FLAGS.separation_model_path,
              min_logit=FLAGS.min_logit,
              target_species=FLAGS.target_species,
              hints_tag=FLAGS.hints_tag,
          )
      )
      # When a file is corrupted and can't be loaded InferenceFn
      # returns None. In this case the lambda below returns false, which then
      # filters it out.
      | beam.Filter(lambda x: x)
      | beam.io.WriteToText(output_prefix, file_name_suffix='.csv')
  )
  pipeline.run()


if __name__ == '__main__':
  app.run(main)
