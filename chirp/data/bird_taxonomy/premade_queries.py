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

"""A set of premade queries to generate stable data configs."""

import json
import logging

from chirp import path_utils
from chirp.data import filter_scrub_utils as fsu
from chirp.taxonomy import namespace_db

# SSW_STATS_PATH contains useful statistics for SSW species, used to guide our
# DFS search in chirp.data.sampling_utils.sample_recordings_under_constraints.
# Those statistics were computed after removing all recordings with foreground
# and background labels belonging to downstream_species.txt
# (mimicking the conditions under which the sampling happens).
DOWNSTREAM_SPECIES_PATH = "data/bird_taxonomy/metadata/downstream_species.txt"
SSW_STATS_PATH = "data/bird_taxonomy/metadata/ssw_stats.json"


def get_upstream_metadata_query() -> fsu.QuerySequence:

  _, _, _, unfeasible_ar_species = get_artificially_rare_species_constraints(
      5, 5)
  db = namespace_db.NamespaceDatabase.load_csvs()
  downstream_species = list(db.class_lists["downstream_species"].classes)
  return fsu.QuerySequence([
      fsu.Query(
          op=fsu.TransformOp.FILTER,
          kwargs={
              "mask_op": fsu.MaskOp.NOT_IN,
              "op_kwargs": {
                  "key": "species_code",
                  "values": downstream_species + unfeasible_ar_species
              }
          })
  ])


def get_artificially_rare_species_constraints(num_foreground: int,
                                              num_background: int):
  """Obtain feasible set of artifically rare species given constraints.

  'Artifically rare' species are species for which we ony want to sample a small
  number of foreground and background recordings. Those species will be useful
  to evaluate long-tail performance of methods. Depending on the exact number
  of fg/bg recordings, some species may not contain enough samples; we call
  those `unfeasible species'. We need to know unfeasible species before-hand
  so that we can (i) remove them for the label space (ii) exclude their
  recordings when searching for a valid solution that satisfies the
  above-mentioned constraints.

  Args:
    num_foreground: The number of foreground recordings we want for each
      species.
    num_background: The number of background recordings we want for each
      species.

  Returns:
    target_fg: The corrected (removing unfeasible species) dictionary of
        foreground constraints.
    target_bg: The corrected (removing unfeasible species) dictionary of
        foreground constraints.
    feasible_ar_species: The set of feasible species.
    unfeasible_ar_species: The set of unfeasible species.
  """
  with open(path_utils.get_absolute_epath(SSW_STATS_PATH), "rb") as f:
    ssw_stats = json.load(f)

  # Fix the target foreground/background for SSW species
  target_fg = {k: num_foreground for k in ssw_stats}
  target_bg = {k: num_background for k in ssw_stats}
  feasible_ar_species = [
      s for s in ssw_stats if ssw_stats[s]["fg"] >= target_fg[s] and
      ssw_stats[s]["bg"] >= target_bg[s]
  ]

  # Re-adjust the target.
  target_fg = {k: num_foreground for k in feasible_ar_species}
  target_bg = {k: num_background for k in feasible_ar_species}

  unfeasible_ar_species = list(
      set(ssw_stats.keys()).difference(feasible_ar_species))
  logging.info(
      "Under constraints (num_foreground=%d, num_background=%d), %d out of %d"
      "SSW species were feasible. The following species were unfeasible: %s",
      num_foreground, num_background, len(feasible_ar_species), len(ssw_stats),
      str(unfeasible_ar_species))

  return target_fg, target_bg, feasible_ar_species, unfeasible_ar_species


def get_upstream_data_query() -> fsu.QuerySequence:
  """Produces the QuerySequence to generate upstream data.

  Returns:
    The QuerySequence to apply
  """
  with open(path_utils.get_absolute_epath(SSW_STATS_PATH), "rb") as f:
    ssw_stats = json.load(f)
  (target_fg, target_bg, feasible_ar_species,
   unfeasible_ar_species) = get_artificially_rare_species_constraints(5, 5)
  db = namespace_db.NamespaceDatabase.load_csvs()
  downstream_species = list(db.class_lists["downstream_species"].classes)

  return fsu.QuerySequence([
      # Filter all samples from downstream species
      fsu.Query(
          op=fsu.TransformOp.FILTER,
          kwargs={
              "mask_op": fsu.MaskOp.NOT_IN,
              "op_kwargs": {
                  "key": "species_code",
                  "values": downstream_species + unfeasible_ar_species
              }
          }),
      # Sample recordings from artificially rare (AR) species.
      fsu.QuerySequence(
          mask_query=fsu.QueryParallel([
              fsu.Query(fsu.MaskOp.CONTAINS_ANY, {
                  "key": "bg_species_codes",
                  "values": feasible_ar_species
              }),
              fsu.Query(fsu.MaskOp.IN, {
                  "key": "species_code",
                  "values": feasible_ar_species
              })
          ], fsu.MergeStrategy.OR),
          queries=[
              # Recall that recordings that contain downstream_species in
              # background (scrubbed for upstream training) are seen during both
              # training and testing. In the meantime, several recordings have
              # both SSW species and downstream species in the background.
              # Therefore, if we allow such a recording to be candidate for
              # sampling, and it ends up being chosen, we will have an AR
              # recording in both upstream and downstream data, which is not
              # good. Hence the filtering op below. Note that ssw_stats contains
              # the statistics of recordings for each AR species **after**
              # filtering out all recordings that contain any downstream
              # species' annotation.
              fsu.Query(
                  op=fsu.TransformOp.FILTER,
                  kwargs={
                      "mask_op": fsu.MaskOp.CONTAINS_NO,
                      "op_kwargs": {
                          "key": "bg_species_codes",
                          "values": downstream_species
                      }
                  }),
              fsu.Query(
                  fsu.TransformOp.SAMPLE_UNDER_CONSTRAINTS, {
                      "species_stats": ssw_stats,
                      "target_fg": target_fg,
                      "target_bg": target_bg
                  })
          ]),
      # Scrub annotations from downstream species
      fsu.Query(
          op=fsu.TransformOp.SCRUB,
          kwargs={
              "key": "bg_species_codes",
              "values": downstream_species + unfeasible_ar_species
          })
  ])


def get_downstream_metadata_query() -> fsu.QuerySequence:

  (_, _, feasible_ar_species,
   _) = get_artificially_rare_species_constraints(5, 5)

  db = namespace_db.NamespaceDatabase.load_csvs()
  downstream_species = list(db.class_lists["downstream_species"].classes)
  return fsu.QuerySequence([
      fsu.Query(
          op=fsu.TransformOp.FILTER,
          kwargs={
              "mask_op": fsu.MaskOp.IN,
              "op_kwargs": {
                  "key": "species_code",
                  "values": downstream_species + feasible_ar_species
              }
          }),
  ])


def get_downstream_data_query() -> fsu.QuerySequence:
  """Produces the QuerySequence to generate downstream data.

  Returns:
    The QuerySequence to apply.
  """
  db = namespace_db.NamespaceDatabase.load_csvs()
  downstream_species = list(db.class_lists["downstream_species"].classes)
  (_, _, feasible_ar_species,
   unfeasible_ar_species) = get_artificially_rare_species_constraints(5, 5)
  upstream_query = get_upstream_data_query()
  return fsu.QuerySequence([
      fsu.QueryComplement(upstream_query, "xeno_canto_id"),
      # We remove unfeasible AR species. For the nominal (5, 5) scenario,
      # this group is empty.
      fsu.Query(
          fsu.TransformOp.FILTER, {
              "mask_op": fsu.MaskOp.NOT_IN,
              "op_kwargs": {
                  "key": "species_code",
                  "values": unfeasible_ar_species
              }
          }),
      # Annotations of species that are not part of the downstream evaluation
      # are scrubbed if they appear in the background or foreground.
      # Therefore, we're only left with relevant species annotated.
      fsu.Query(
          op=fsu.TransformOp.SCRUB_ALL_BUT,
          kwargs={
              "key": "bg_species_codes",
              "values": downstream_species + feasible_ar_species
          }),
      fsu.Query(
          op=fsu.TransformOp.SCRUB_ALL_BUT,
          kwargs={
              "key": "species_code",
              "values": downstream_species + feasible_ar_species,
          })
  ])
