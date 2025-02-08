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

"""Call density estimation using a bernoulli kernel."""

from flax import nnx
import jax
from jax import numpy as jnp


def log_sum_exp(xs: jnp.ndarray, axis=None):
  # Log-of-sum-of-exponentials admits a nice, more-stable form using the max
  # of the sequence.
  # https://mc-stan.org/docs/2_27/stan-users-guide/log-sum-of-exponentials.html
  max_x = jnp.max(xs, axis=axis, keepdims=True)
  max_sq = jnp.max(xs, axis=axis)
  sums_x = jnp.log(jnp.sum(jnp.exp(xs - max_x), axis=axis))
  return sums_x + max_sq


sq_norm = lambda x: jnp.sum(x * x, axis=1)
dots_ab = lambda x, y: jnp.dot(x, y.T)
dists_ab = lambda x, y: (
    -2 * dots_ab(x, y) + sq_norm(x)[:, jnp.newaxis] + sq_norm(y)[jnp.newaxis, :]
)

# Scaled distances.
dists_ab_s = lambda a, b, s: (
    dists_ab(a * s[jnp.newaxis, :], b * s[jnp.newaxis, :])
)


def scaled_rbf_kernel(
    x: jnp.ndarray, y: jnp.ndarray, scale: jnp.ndarray, bias: float
):
  return dists_ab_s(x, y, scale) + bias


class BernoulliData(nnx.Variable):
  """Container for data and labels for a BernoulliProcessor."""

  # We declare this subclass so that the groundtruth data is not updated
  # during training.
  pass


class BernoulliRBF(nnx.Module):
  r"""Model P(+|x) ~ \beta(a(x), b(x)).

  Given some input data x, we want to estimate the number of virtual positive
  and negative observations to associate with x. These are used as parameters in
  a beta distribution, allowing us to have both an expected value for P(+|x)
  and a measure of certainty, according to the total weight a(x) + b(x).

  We combine two approaches for estimating a(x), b(x):
  First, a learned RBF kernel over the ground-truth observations acts as a KNN
  classifier, contributing positive and negative observations at arbitrary x
  according to learned similarity between x and the groundtruth.

  Second, we (optionally) directly predict a number of pos/neg observations
  a_f(x), b_f(x) from the features themselves. For example, if one of the
  features is a classifier score, this allows the model to directly use the
  classifier score as a prior, with some learned weight.
  """

  def __init__(
      self,
      data: jnp.ndarray,
      data_labels: jnp.ndarray,
      data_mean: float | None = 0.0,
      data_std: float | None = 1.0,
      learn_feature_weights: bool = False,
      *,
      rngs: nnx.Rngs
  ):
    key = rngs.params()
    num_features = data.shape[-1]
    self.scales_pos = nnx.Param(jax.random.uniform(key, (num_features,)))
    self.scales_neg = nnx.Param(jax.random.uniform(key, (num_features,)))
    self.weight_bias = nnx.Param(jnp.zeros([2]))
    if data_mean is None:
      self.data_mean = BernoulliData(jnp.mean(data, axis=0, keepdims=True))
    else:
      self.data_mean = BernoulliData(data_mean)
    if data_std is None:
      self.data_stds = BernoulliData(jnp.std(data, axis=0, keepdims=True))
    else:
      self.data_stds = BernoulliData(data_std)
    data_pos, data_neg = self.split_labeled_data(data, data_labels)
    self.data_pos = jax.lax.stop_gradient(
        BernoulliData(self._normalize(data_pos))
    )
    self.data_neg = jax.lax.stop_gradient(
        BernoulliData(self._normalize(data_neg))
    )
    self.data_labels = jax.lax.stop_gradient(BernoulliData(data_labels))
    self.learn_feature_weights = learn_feature_weights

    # Matrices for assigning pos/neg weight directly from features.
    self.feature_weights = nnx.Param(jax.random.uniform(key, (num_features, 2)))
    self.feature_bias = nnx.Param(jax.random.uniform(key, (2,)))

  @classmethod
  def split_labeled_data(cls, data: jnp.ndarray, data_labels: jnp.ndarray):
    pos_idxes = jnp.where(data_labels == 1)[0]
    neg_idxes = jnp.where(data_labels == 0)[0]
    data_pos = data[pos_idxes]
    data_neg = data[neg_idxes]
    return data_pos, data_neg

  def _normalize(self, x):
    return (x - self.data_mean.value) / self.data_stds.value

  def _log_counts(self, x: jnp.ndarray, normalize: bool = True):
    if normalize:
      x = self._normalize(x)
    pos_count = scaled_rbf_kernel(
        x, self.data_pos, self.scales_pos, self.weight_bias[0]
    )
    neg_count = scaled_rbf_kernel(
        x, self.data_neg, self.scales_neg, self.weight_bias[1]
    )

    if self.learn_feature_weights:
      feature_count = jnp.dot(x, self.feature_weights.value) + self.feature_bias
      pos_count = jnp.concat([pos_count, feature_count[:, :1]], axis=1)
      neg_count = jnp.concat([neg_count, feature_count[:, 1:]], axis=1)
    log_pos_count = log_sum_exp(-pos_count, axis=1)
    log_neg_count = log_sum_exp(-neg_count, axis=1)
    log_weight_count = log_sum_exp(
        jnp.concatenate([-pos_count, -neg_count], axis=1), axis=1
    )
    return log_pos_count, log_neg_count, log_weight_count

  def __call__(self, x: jnp.ndarray, normalize: bool = True):
    """Compute log(P(+|x)) and the total example weight of x."""
    log_pos_count, _, log_weight_count = self._log_counts(x, normalize)
    log_p_x = log_pos_count - log_weight_count
    return log_p_x, log_weight_count

  def sampled_counts(self, seed: int, x: jnp.ndarray, n_samples: int = 1024):
    """Create sampled positive counts from the learned distribution at x."""
    log_pos_count, log_neg_count, unused_log_wt = self._log_counts(x)
    pos_count = jnp.exp(log_pos_count)[:, jnp.newaxis]
    neg_count = jnp.exp(log_neg_count)[:, jnp.newaxis]

    k = jax.random.PRNGKey(seed)
    beta_samp = jax.random.beta(
        k, pos_count, neg_count, shape=[pos_count.shape[0], n_samples]
    )
    sample_counts = jnp.sum(
        jax.random.uniform(k, shape=beta_samp.shape) < beta_samp, axis=0
    )
    return sample_counts

  def gt_log_likelihood(self):
    """Total log likelihood of the GT data, given learned params."""
    # Counts for positive points.
    pos_pos_count = scaled_rbf_kernel(  # [N+, N+]
        self.data_pos, self.data_pos, self.scales_pos, self.weight_bias[0]
    )
    pos_neg_count = scaled_rbf_kernel(
        self.data_pos, self.data_neg, self.scales_neg, self.weight_bias[1]
    )

    # Counts for negative points.
    neg_neg_count = scaled_rbf_kernel(
        self.data_neg, self.data_neg, self.scales_neg, self.weight_bias[1]
    )
    neg_pos_count = scaled_rbf_kernel(
        self.data_neg, self.data_pos, self.scales_pos, self.weight_bias[0]
    )

    # Estimate pos/neg priors from raw features.
    if self.learn_feature_weights:
      pos_feature_count = (
          jnp.dot(self.data_pos.value, self.feature_weights.value)  # [N+, 2]
          + self.feature_bias.value
      )
      neg_feature_count = (
          jnp.dot(self.data_neg.value, self.feature_weights.value)  # [N-, 2]
          + self.feature_bias.value
      )
      # Add feature counts to the list of actual counts from data.
      pos_pos_count = jnp.concat(
          [pos_pos_count, pos_feature_count[:, :1]], axis=-1
      )
      pos_neg_count = jnp.concat(
          [pos_neg_count, pos_feature_count[:, 1:]], axis=-1
      )
      neg_pos_count = jnp.concat(
          [neg_pos_count, neg_feature_count[:, :1]], axis=-1
      )
      neg_neg_count = jnp.concat(
          [neg_neg_count, neg_feature_count[:, 1:]], axis=-1
      )

    pos_log_prob = log_sum_exp(-pos_pos_count, axis=1) - log_sum_exp(
        jnp.concatenate([-pos_pos_count, -pos_neg_count], axis=1), axis=1
    )
    neg_log_prob = log_sum_exp(-neg_neg_count, axis=1) - log_sum_exp(
        jnp.concatenate([-neg_neg_count, -neg_pos_count], axis=1), axis=1
    )

    pos_log_prob = jnp.mean(pos_log_prob)
    neg_log_prob = jnp.mean(neg_log_prob)
    return pos_log_prob + neg_log_prob

  def matching_loss(self):
    """Difference between observed log P(+) and estimated log P(+)."""
    data_log_p_x, _ = self(
        jnp.concatenate([self.data_pos, self.data_neg], axis=0), normalize=False
    )
    target_log_p_x = jnp.log(self.data_pos.shape[0]) - jnp.log(
        self.data_pos.shape[0] + self.data_neg.shape[0]
    )
    return jnp.abs(data_log_p_x.mean() - target_log_p_x)


@nnx.jit
def train_step(
    model: BernoulliRBF, optimizer: nnx.optimizer.Optimizer, mu: float
) -> float:
  def loss_fn(model: BernoulliRBF):
    gt_log_likelihood_loss = -model.gt_log_likelihood()
    matching_loss = model.matching_loss()
    return gt_log_likelihood_loss + mu * matching_loss

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)
  return loss
