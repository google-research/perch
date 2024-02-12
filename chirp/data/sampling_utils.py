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

"""Utilities for subsampling dataframes."""

import bisect
import collections
import copy
import hashlib


from absl import logging
import jax
import numpy as np
import pandas as pd

# The way a recording is represented in the DFS carried out in
# `sample_recordings_under_constraints`. First element is its foreground
# species. Second element is its list of background species. Final element is
# its index in the dataframe.
_RECORDING = tuple[str, list[str], int]


def sample_recordings(
    df: pd.DataFrame,
    target_fg: dict[str, int],
    prng_seed: int,
):
  """Subsamples recordings from df.

  Args:
    df: The dataframe to subsample.
    target_fg: A dictionnary mapping each species to its required number of
      foreground recordings to be subsampled.
    prng_seed: The PRNG seed to use for random sampling.

  Returns:
    The subsampled df such that there are exactly target_fg[species] foreground
    labels of each species.
  """
  key = jax.random.PRNGKey(prng_seed)

  def _subsample(group_df):
    (species_code,) = group_df['species_code'].unique()
    indices = sorted(
        jax.random.choice(
            # Create a unique key derived from the global key and the species
            # code.
            jax.random.fold_in(
                key,
                int(
                    hashlib.md5(species_code.encode()).hexdigest()[:8], base=16
                ),
            ),
            len(group_df),
            shape=(target_fg[species_code],),
            replace=False,
        ).tolist()
    )
    return group_df.iloc[indices]

  return (
      df[df['species_code'].isin(target_fg.keys())]
      .groupby('species_code', group_keys=False)
      .apply(_subsample)
  )


def hit_target(count_dic: dict[str, int]) -> bool:
  return np.all(np.array(list(count_dic.values())) == 0)


def find_valid_subset(
    remaining_fg: collections.OrderedDict[str, int],
    remaining_bg: collections.OrderedDict[str, int],
    chosen: list[_RECORDING],
    seen: dict[tuple[_RECORDING], bool],
    candidates: list[_RECORDING],
) -> list[_RECORDING] | None:
  """Function that tries to find a valid solution to sampling under constraints.

  This function performs a DFS to find a subset of recordings that satisfies
  the constraints. The list of `chosen` recordings defines the current path
  in the tree. Because randomly searching for solutions is generally
  intractable, we guide the search by only solving the constraints for one
  species at a time. The order in which species are addressed is implicitly
  defined by the order of the keys from remaining_fg and remaining_bg (should)
  coincide.

  Args:
    remaining_fg: For each species, the number of foreground recordings left to
      find. Species (keys) should ideally be sorted from hardest to easiest.
    remaining_bg: For each species, the number of background recordings left to
      find. Species (keys) should ideally be sorted from hardest to easiest.
    chosen: The recordings chosen so far.
    seen: The branches of the tree already visited.
    candidates: The pool of recordings to pick from.

  Returns:
    The solution (=list of recordings) if it finds any, None otherwise.
  """
  # Check if we hit the target
  if hit_target(remaining_fg) and hit_target(remaining_bg):
    return chosen

  # Check that we still have candidates
  if not candidates:
    return None

  # Check that we haven't already been there. `chosen` needs to be sorted
  # for this to work. This is ensured when we construct `chosen`.
  if seen[tuple(chosen)]:
    return None

  # Else we continue visiting. We focus on a single species at a time. We
  # fetch the first species for which the constraints are not yet satisfied.
  # In the event that remaining_bg's keys are sorted by decreasing difficulty,
  # this corresponds to fetching the most difficult species not yet satisfied.
  for s in remaining_bg:
    if remaining_bg[s] > 0 or remaining_fg[s] > 0:
      current_species = s
      break
  for index, recording in enumerate(candidates):
    if valid_recording(recording, remaining_fg, remaining_bg, current_species):
      updated_fg = copy.copy(remaining_fg)
      updated_bg = copy.copy(remaining_bg)
      if recording[0] in updated_fg:
        updated_fg[recording[0]] -= 1
      # In background, a same species may appear twice, so we take set() to
      # remove dupplicates.
      for bg_rec in set(recording[1]):
        if bg_rec in updated_bg:
          updated_bg[bg_rec] -= 1
      new_chosen = copy.copy(chosen)
      bisect.insort(new_chosen, recording)
      res = find_valid_subset(
          updated_fg,
          updated_bg,
          new_chosen,
          seen,
          [x for i, x in enumerate(candidates) if i != index],
      )
      if res is not None:
        return res
  seen[tuple(chosen)] = True
  return None


def valid_recording(
    recording: _RECORDING,
    remaining_fg: collections.OrderedDict[str, int],
    remaining_bg: collections.OrderedDict[str, int],
    current_species: str,
) -> bool:
  """Decides whether a child (=recording) should be explored next.

  The function checks whether (i) The recording contains the species we are
  currently addressing (ii) if yes, whether adding this recording to 'chosen'
  wouldn't violate any constraint.

  Args:
    recording: The recording whose relevance we want to assess.
    remaining_fg: For each species, the number of foreground recordings left to
      find. Species (keys) should ideally be sorted from hardest to easiest.
    remaining_bg: For each species, the number of background recordings left to
      find. Species (keys) should ideally be sorted from hardest to easiest.
    current_species: The current species the search is focused on satisfying.

  Returns:
    True if the recording should be explored, False otherwise.
  """
  # Ensure the current_species is in this recording.
  if (
      remaining_fg[current_species] > 0 and recording[0] == current_species
  ) or (remaining_bg[current_species] > 0 and current_species in recording[1]):
    # Ensure it doesn't violate any constraint.
    violates_fg = (
        recording[0] in remaining_fg and remaining_fg[recording[0]] == 0
    )
    violates_bg = any(
        [x in remaining_bg and remaining_bg[x] == 0 for x in recording[1]]
    )
    if not violates_fg and not violates_bg:
      return True
  return False
