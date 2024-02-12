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

"""A set of premade queries to generate stable data configs."""

from chirp.data import filter_scrub_utils as fsu
from chirp.taxonomy import namespace_db

# SSW_STATS_PATH contains useful statistics for SSW species, used to guide our
# DFS search in chirp.data.sampling_utils.sample_recordings_under_constraints.
# Those statistics were computed after removing all recordings with foreground
# and background labels belonging to downstream_species.txt
# (mimicking the conditions under which the sampling happens).
DOWNSTREAM_SPECIES_PATH = "data/bird_taxonomy/metadata/downstream_species.txt"
SSW_STATS_PATH = "data/bird_taxonomy/metadata/ssw_stats.json"

DOWNSTREAM_CLASS_LIST = "downstream_species_v2"
AR_CLASS_LIST = "artificially_rare_species_v2"
AR_SAMPLING_PRNG_SEED = 2023 + 5 + 11


def get_upstream_metadata_query() -> fsu.QuerySequence:
  db = namespace_db.load_db()
  downstream_species = list(db.class_lists[DOWNSTREAM_CLASS_LIST].classes)
  return fsu.QuerySequence(
      [
          fsu.Query(
              op=fsu.TransformOp.FILTER,
              kwargs={
                  "mask_op": fsu.MaskOp.NOT_IN,
                  "op_kwargs": {
                      "key": "species_code",
                      "values": downstream_species,
                  },
              },
          )
      ]
  )


def get_upstream_data_query(ar_only: bool = False) -> fsu.QuerySequence:
  """Produces the QuerySequence to generate upstream data.

  Args:
    ar_only: if True, only include recordings with artificially rare species
      annotations.

  Returns:
    The QuerySequence to apply
  """
  db = namespace_db.load_db()
  downstream_species = list(db.class_lists[DOWNSTREAM_CLASS_LIST].classes)
  # NOTE: Artificially rare species are the subset of SSW species which do not
  # intersect with the downstream species.
  ar_species = list(db.class_lists[AR_CLASS_LIST].classes)

  queries = [
      # Filter all samples from downstream species
      fsu.Query(
          op=fsu.TransformOp.FILTER,
          kwargs={
              "mask_op": fsu.MaskOp.NOT_IN,
              "op_kwargs": {
                  "key": "species_code",
                  "values": downstream_species,
              },
          },
      ),
      # Sample recordings from artificially rare (AR) species.
      fsu.QuerySequence(
          mask_query=fsu.Query(
              fsu.MaskOp.IN,
              {"key": "species_code", "values": ar_species},
          ),
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
                          "values": downstream_species,
                      },
                  },
              ),
              fsu.Query(
                  fsu.TransformOp.SAMPLE,
                  {
                      "target_fg": {k: 10 for k in ar_species},
                      "prng_seed": AR_SAMPLING_PRNG_SEED,
                  },
              ),
          ],
      ),
      # Scrub annotations from downstream species
      fsu.Query(
          op=fsu.TransformOp.SCRUB,
          kwargs={
              "key": "bg_species_codes",
              "values": downstream_species + ar_species,
          },
      ),
  ]
  if ar_only:
    queries.append(
        fsu.Query(
            op=fsu.TransformOp.FILTER,
            kwargs={
                "mask_op": fsu.MaskOp.IN,
                "op_kwargs": {
                    "key": "species_code",
                    "values": ar_species,
                },
            },
        )
    )

  return fsu.QuerySequence(queries)


def get_downstream_metadata_query() -> fsu.QuerySequence:
  db = namespace_db.load_db()
  downstream_species = list(db.class_lists[DOWNSTREAM_CLASS_LIST].classes)
  # NOTE: Artificially rare species are the subset of SSW species which do not
  # intersect with the downstream species.
  ar_species = list(db.class_lists[AR_CLASS_LIST].classes)
  return fsu.QuerySequence([
      fsu.Query(
          op=fsu.TransformOp.FILTER,
          kwargs={
              "mask_op": fsu.MaskOp.IN,
              "op_kwargs": {
                  "key": "species_code",
                  "values": downstream_species + ar_species,
              },
          },
      ),
  ])


def get_downstream_data_query() -> fsu.QuerySequence:
  """Produces the QuerySequence to generate downstream data.

  Returns:
    The QuerySequence to apply.
  """
  db = namespace_db.load_db()
  downstream_species = list(db.class_lists[DOWNSTREAM_CLASS_LIST].classes)
  # NOTE: Artificially rare species are the subset of SSW species which do not
  # intersect with the downstream species.
  ar_species = list(db.class_lists[AR_CLASS_LIST].classes)
  upstream_query = get_upstream_data_query()
  return fsu.QuerySequence([
      fsu.QueryComplement(upstream_query, "xeno_canto_id"),
      # Annotations of species that are not part of the downstream evaluation
      # are scrubbed if they appear in the background or foreground.
      # Therefore, we're only left with relevant species annotated.
      fsu.Query(
          op=fsu.TransformOp.SCRUB_ALL_BUT,
          kwargs={
              "key": "bg_species_codes",
              "values": downstream_species + ar_species,
          },
      ),
      fsu.Query(
          op=fsu.TransformOp.SCRUB_ALL_BUT,
          kwargs={
              "key": "species_code",
              "values": downstream_species + ar_species,
          },
      ),
  ])


def get_class_representatives_metadata_query() -> fsu.QuerySequence:
  db = namespace_db.load_db()
  species = list(
      set(
          list(db.class_lists[DOWNSTREAM_CLASS_LIST].classes)
          + list(db.class_lists[AR_CLASS_LIST].classes)
          + list(db.class_lists["high_sierras"].classes)
          + list(db.class_lists["sierras_kahl"].classes)
          + list(db.class_lists["peru"].classes)
      )
  )
  return fsu.QuerySequence(
      [
          fsu.Query(
              op=fsu.TransformOp.FILTER,
              kwargs={
                  "mask_op": fsu.MaskOp.IN,
                  "op_kwargs": {
                      "key": "species_code",
                      "values": species,
                  },
              },
          )
      ]
  )


def get_class_representatives_data_query() -> fsu.QuerySequence:
  """Produces the QuerySequence to generate class representatives data."""
  db = namespace_db.load_db()
  species = list(
      set(
          list(db.class_lists[DOWNSTREAM_CLASS_LIST].classes)
          + list(db.class_lists["high_sierras"].classes)
          + list(db.class_lists["sierras_kahl"].classes)
          + list(db.class_lists["peru"].classes)
      )
  )
  species_no_ar = [
      s for s in species if s not in db.class_lists[AR_CLASS_LIST].classes
  ]
  return fsu.QuerySequence([
      fsu.QueryParallel(
          queries=[
              fsu.Query(
                  op=fsu.TransformOp.FILTER,
                  kwargs={
                      "mask_op": fsu.MaskOp.IN,
                      "op_kwargs": {
                          "key": "species_code",
                          "values": species_no_ar,
                      },
                  },
              ),
              get_upstream_data_query(ar_only=True),
          ],
          merge_strategy=fsu.MergeStrategy.CONCAT_NO_DUPLICATES,
      ),
      # This scrubs all background labels except those with values in `species`.
      fsu.Query(
          op=fsu.TransformOp.SCRUB_ALL_BUT,
          kwargs={
              "key": "bg_species_codes",
              "values": species,
          },
      ),
  ])
