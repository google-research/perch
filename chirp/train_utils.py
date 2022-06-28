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

"""Utility functions for training loops."""

import os
from typing import Any, Optional, Sequence

from absl import logging
from clu import checkpoint
import jax


def normalize_checkpoint_path(ckpt_path):
  if ckpt_path.endswith('-0'):
    ckpt_path = ckpt_path[:-2]
  if not ckpt_path.startswith('/readahead'):
    ckpt_path = os.path.join('/readahead/256M', ckpt_path)
  return ckpt_path


def _validate_matching_shapes(tree_left, tree_right):
  """Check that inputs have the same structure and array shapes."""
  if jax.tree_structure(tree_left) != jax.tree_structure(tree_right):
    raise ValueError('Incompatible param structure in warmstart_checkpoint.')

  shapes_match = jax.tree_map(lambda x, y: x.shape == y.shape, tree_left,
                              tree_right)
  if not jax.tree_util.tree_all(shapes_match):
    logging.error('Warmstart checkpoint contains mismatched shapes: %s',
                  shapes_match)
    raise ValueError('Incompatible param shapes in warmstart_checkpoint.')


def selective_warmstart(train_state: Any,
                        ckpt_path: str,
                        keys: Optional[Sequence[str]] = None):
  """Loads model params from ckpt_path into train_state.params.

  Args:
    train_state: Dataclass template. The model weights should be in
      train_state.params.
    ckpt_path: Path containing model checkpoint to restore from.
    keys: Optional list of sub-trees to restore.

  Returns:
    Updated train_state with restored params.
  Raises:
    ValueError if there are mismatched structures or shapes in desired params.
  """
  ckpt = checkpoint.MultihostCheckpoint(ckpt_path)
  if not keys:
    restored = ckpt.restore(train_state)
    _validate_matching_shapes(train_state.params, restored.params)
    new_params = restored.params
  else:
    # With no state template, restored is a dict.
    restored = ckpt.restore(state=None)
    new_params = train_state.params.unfreeze()
    for key in keys:
      _validate_matching_shapes(new_params[key], restored['params'][key])
      new_params[key] = restored['params'][key]
  return train_state.replace(params=new_params)
