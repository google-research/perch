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

"""Per-Channel Energy Normalization (PCEN) ops.

See https://arxiv.org/abs/1607.05666 for details.
"""

import tensorflow

tf = tensorflow.compat.v1


def _swap_initial_and_time_axes(tensor):
  """Swaps the initial axis with the one that index STFT frame index.

  This assumes that the axis at index -2 indexes STFT frames. The
  tf.while_loop
  in the PCEN Op will want to iterate over that axis for the smoothing filter
  and for that also wants it to be axis 0. This method is intended to be
  applied
  before and after the while_loop and just swaps those two axes.

  Args:
    tensor: A tensor for which the axis to be smoothed over is at index -2. It
      is expected but not required that its rank will be in {2, 3}.

  Returns:
    Transpose of tensor where axes (0, -2) have been swapped.
  """
  if tensor.shape.rank is not None:
    if tensor.shape.rank < 3:
      return tensor

    perm = list(range(tensor.shape.rank))
    perm[0], perm[-2] = perm[-2], perm[0]
    return tf.transpose(tensor, perm)

  rank = tf.rank(tensor)

  def return_original_tensor():
    return tensor

  def return_permuted_tensor():
    perm = tf.range(rank)
    # Overly complex way of swapping element 0 and -2.
    perm = tf.concat([perm[-2:-1], perm[1:-2], perm[0:1], perm[-1:]], axis=0)
    # It appears that, even when rank < 3, this path must still be valid. When
    # rank < 3, the previous line will add an element to the perm list.
    perm = perm[0:rank]
    return tf.transpose(tensor, perm)

  return tf.cond(
      rank < 3, true_fn=return_original_tensor, false_fn=return_permuted_tensor
  )


def fixed_pcen(
    filterbank_energy,
    alpha,
    smooth_coef,
    delta=2.0,
    root=2.0,
    floor=1e-6,
    name=None,
    streaming=False,
    state=None,
):
  """Per-Channel Energy Normalization (PCEN) with fixed parameters.

  See https://arxiv.org/abs/1607.05666 for details.

  Args:
    filterbank_energy: A [..., num_frames, num_frequency_bins] tensor of
      power-domain filterbank energies. If a scalar, we return 0.0 as the
      spectral floor value (for padding purposes).
    alpha: The normalization coefficient.
    smooth_coef: The coefficient of the IIR smoothing filter ($s$ in the paper).
    delta: Constant stabilizer offset for the root compression.
    root: Root compression coefficient.
    floor: Epsilon floor value to prevent division by zero.
    name: Optional scope name.
    streaming: If true, also return a smoothing output so that this function can
      be run on sequential chunks of audio, instead of processing all audio at
      once.
    state: Optional state produced by a previous call to fixed_pcen. Used in
      streaming mode.

  Returns:
    Filterbank energies with PCEN compression applied (type and shape are
    unchanged). If in streaming mode, also returns a state tensor to be used
    in the next call to fixed_pcen.
  """
  with tf.name_scope(name, 'pcen'):
    filterbank_energy = tf.convert_to_tensor(filterbank_energy)
    if filterbank_energy.shape.rank == 0:
      return tf.constant(0.0, filterbank_energy.dtype)
    filterbank_energy.shape.with_rank_at_least(2)

    alpha = tf.convert_to_tensor(alpha, filterbank_energy.dtype, name='alpha')
    alpha.shape.with_rank_at_most(1)
    smooth_coef = tf.convert_to_tensor(
        smooth_coef, filterbank_energy.dtype, name='smoothing_coefficient'
    )
    smooth_coef.shape.assert_has_rank(0)
    delta = tf.convert_to_tensor(delta, filterbank_energy.dtype, name='delta')
    delta.shape.with_rank_at_most(1)
    root = tf.convert_to_tensor(root, filterbank_energy.dtype, name='root')
    root.shape.with_rank_at_most(1)
    floor = tf.convert_to_tensor(floor, filterbank_energy.dtype, name='floor')
    floor.shape.assert_has_rank(0)

    # Compute the smoothing filter.
    transposed_energy = _swap_initial_and_time_axes(filterbank_energy)
    timesteps = tf.shape(transposed_energy)[0]
    filterbank_energy_ta = tf.TensorArray(
        filterbank_energy.dtype, size=timesteps, clear_after_read=False
    )
    filterbank_energy_ta = filterbank_energy_ta.unstack(transposed_energy)

    def compute_smoother():
      """Compute a first-order smoothing filter."""

      if state is not None:
        init_smoother = state
      else:
        init_smoother = filterbank_energy_ta.read(0)

      def _cond(t, unused_smoother_ta, unused_prev_ret):
        return t < timesteps

      def _body(t, smoother_ta, prev_ret):
        cur_ret = (
            1.0 - smooth_coef
        ) * prev_ret + smooth_coef * filterbank_energy_ta.read(t)
        smoother_ta = smoother_ta.write(t, cur_ret)
        return t + 1, smoother_ta, cur_ret

      smoother_ta = tf.TensorArray(
          filterbank_energy.dtype, timesteps, clear_after_read=False
      )
      _, smoother_ta, final_smoother = tf.while_loop(
          _cond,
          _body,
          loop_vars=[tf.constant(0, tf.int32), smoother_ta, init_smoother],
      )
      return _swap_initial_and_time_axes(smoother_ta.stack()), final_smoother

    smoother, final_state = compute_smoother()

    one_over_root = 1.0 / root
    pcen = (
        filterbank_energy / (floor + smoother) ** alpha + delta
    ) ** one_over_root - delta**one_over_root

    if streaming:
      return pcen, final_state
    else:
      return pcen
