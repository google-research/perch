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

"""Tooling for measuring call density."""

import collections
import dataclasses
from typing import Optional

from chirp.inference.search import search
from etils import epath
import ipywidgets
import numpy as np
import pandas as pd
import scipy


VALIDATION_FILE_PATTERN = 'validation_%s.csv'


@dataclasses.dataclass
class ValidationExample:
  """Wrapper for validation data used for call density estimation.

  Attributes:
    filename: Source file for the validation example.
    timestamp_offset: Offset within the source file.
    score: Classifier or search score.
    is_pos: 1 if a positive, -1 if negative, 0 if unknown.
    bin: Bin number the valdiation example was assigned to.
    bin_weight: Proportion of total data in the same bin as this example.
  """

  filename: str
  timestamp_offset: float
  score: float
  is_pos: int
  bin: int
  bin_weight: float

  def to_row(self):
    return [
        self.filename,
        self.timestamp_offset,
        self.score,
        self.is_pos,
        self.bin,
        self.bin_weight,
    ]

  def to_search_result(self, target_class: str):
    """Convert to a search result for display only."""
    result = search.SearchResult(
        filename=self.filename,
        timestamp_offset=self.timestamp_offset,
        score=self.score,
        sort_score=np.random.uniform(),
        embedding=np.zeros(shape=(0,), dtype=np.float32),
    )
    b = ipywidgets.RadioButtons(
        options=[target_class, f'not {target_class}', 'unsure']
    )
    if self.is_pos == 1:
      b.value = target_class
    elif self.is_pos == -1:
      b.value = f'not {target_class}'
    elif self.is_pos == 0:
      b.value = 'unsure'
    else:
      raise ValueError(f'unexpected value ({self.is_pos})')
    result.label_widgets = [b]
    return result

  @classmethod
  def from_search_result(
      cls,
      result: search.SearchResult,
      target_class: str,
      quantile_bounds: np.ndarray,
      value_bounds: np.ndarray,
  ) -> Optional['ValidationExample']:
    """Create a ValidationExample from a search result."""
    bin_weights = quantile_bounds[1:] - quantile_bounds[:-1]
    if not result.label_widgets:
      return None
    value = result.label_widgets[0].value
    if value is None:
      return None
    if value == target_class:
      is_pos = 1
    elif value == f'not {target_class}':
      is_pos = -1
    elif value == 'unsure':
      is_pos = 0
    else:
      raise ValueError(f'unexpected value ({value})')
    result_bin = max(np.argmax(result.score < value_bounds) - 1, 0)
    return ValidationExample(
        filename=result.filename,
        timestamp_offset=result.timestamp_offset,
        score=result.score,
        is_pos=is_pos,
        bin=result_bin,
        bin_weight=bin_weights[result_bin],
    )


def prune_random_results(
    results: search.TopKSearchResults,
    all_scores: np.ndarray,
    quantile_bounds: np.ndarray,
    samples_per_bin: int,
) -> search.TopKSearchResults:
  """Reduce results to at most samples_per_bin examples per quantile bin."""
  value_bounds = np.quantile(all_scores, quantile_bounds)
  binned = [[] for _ in range(quantile_bounds.shape[0] - 1)]
  for r in results.search_results:
    result_bin = max(np.argmax(r.score < value_bounds) - 1, 0)
    binned[result_bin].append(r)
  binned = [np.random.choice(b, samples_per_bin, replace=False) for b in binned]

  combined_results = []
  for b in binned:
    combined_results.extend(b)
  rng = np.random.default_rng(42)
  rng.shuffle(combined_results)
  return search.TopKSearchResults(len(combined_results), combined_results)


def convert_combined_results(
    combined_results: search.TopKSearchResults,
    target_class: str,
    quantile_bounds: np.ndarray,
    value_bounds: np.ndarray,
) -> list[ValidationExample]:
  """Convert a TopKSearchResults to a list of ValidationExamples."""
  examples = []
  for r in combined_results:
    ex = ValidationExample.from_search_result(
        r, target_class, quantile_bounds, value_bounds
    )
    if ex is not None:
      examples.append(ex)
  return examples


def get_random_sample_size(quantile_bounds: np.ndarray, samples_per_bin: int):
  """Estimate random samples needed to get the target samples_per_bin."""
  bin_weights = quantile_bounds[1:] - quantile_bounds[:-1]
  rarest_weight = np.min(bin_weights)
  return int(2 * samples_per_bin / rarest_weight)


def write_validation_log(
    validation_examples: list[ValidationExample],
    output_path: epath.Path,
    target_class: str,
) -> epath.Path:
  """Write a set of validation results to a log file."""
  validation_log_filepath = epath.Path(output_path) / (
      VALIDATION_FILE_PATTERN % target_class
  )
  # Deduplicate examples, preferring new instances.
  examples = {}
  if validation_log_filepath.exists():
    existing_examples = load_validation_log(validation_log_filepath)
    for ex in existing_examples:
      examples[(ex.filename, ex.timestamp_offset)] = ex
  for ex in validation_examples:
    examples[(ex.filename, ex.timestamp_offset)] = ex

  write_examples = list(examples.values())
  df = pd.DataFrame(write_examples)
  fieldnames = [f.name for f in dataclasses.fields(ValidationExample)]
  df.to_csv(validation_log_filepath, mode='w', columns=fieldnames, index=False)
  return validation_log_filepath


def load_validation_log(validation_log_filepath) -> list[ValidationExample]:
  """From the log CSV, load a sequence of ValidationExamples."""
  df = pd.read_csv(validation_log_filepath)
  data = []
  fields = dataclasses.fields(ValidationExample)
  type_map = {f.name: f.type for f in fields}
  for _, row in df.iterrows():
    row = {k: type_map[k](v) for k, v in row.to_dict().items()}
    data.append(ValidationExample(**row))
  return data


def estimate_call_density(
    examples: list[ValidationExample],
    num_beta_samples: int = 10_000,
    beta_prior: float = 0.1,
) -> tuple[float, np.ndarray]:
  """Estimates call density from a set of ValidationExample.

  Args:
    examples: Validated examples.
    num_beta_samples: Number of times to draw from beta distributions.
    beta_prior: Prior for beta distribution.

  Returns:
    Expected value of density and an array of all sampled density estimates.
  """
  # Collect validated labels by bin.
  bin_pos = collections.defaultdict(int)
  bin_neg = collections.defaultdict(int)
  bin_weights = collections.defaultdict(float)
  for ex in examples:
    bin_weights[ex.bin] = ex.bin_weight
    if ex.is_pos == 1:
      bin_pos[ex.bin] += 1
    elif ex.is_pos == -1:
      bin_neg[ex.bin] += 1

  # Create beta distributions.
  betas = {}
  for b in bin_weights:
    betas[b] = scipy.stats.beta(
        bin_pos[b] + beta_prior, bin_neg[b] + beta_prior
    )

  # MLE positive rate in each bin.
  density_ev = np.array([
      bin_weights[b] * bin_pos[b] / (bin_pos[b] + bin_neg[b] + 1e-6)
      for b in bin_weights
  ]).sum()

  q_betas = []
  for _ in range(num_beta_samples):
    q_beta = np.array([
        bin_weights[b] * betas[b].rvs(size=1)[0] for b in betas  # p(b) * P(+|b)
    ]).sum()
    q_betas.append(q_beta)

  return density_ev, np.array(q_betas)


def estimate_roc_auc(
    examples: list[ValidationExample],
) -> float:
  """Estimate the classifier ROC-AUC from validation logs.

  We use the probabalistic interpretation of ROC-AUC, as the probability that
  a uniformly sampled positive example has higher score than a uniformly sampled
  negative example.

  Abusing notation a bit, we decompose this as follows:
  P(+ > -) = sum_{b,c} P(+ > - | + in b, - in c) * P(b|+) * P(c|-)
  using the law of total probability and independence of drawing the pos/neg
  examples uniformly at random.

  When comparing scores from different bins b > c, we have all scores from b
  greater than scores from c, so P(+ > - | + in b, - in c) = 1.
  Likewise, if b < c, P(+ > - | + in b, - in c) = 0.

  When b == c, we can count directly the number of + > - pairs to estimate the
  in-bin ROC-AUC.

  The scalars P(b|+) and P(c|-) are computed using Bayes' rule and the expected
  value estimate of P(+).

  Args:
    examples: List of ValidationExample objects.

  Returns:
    ROC-AUC estimate.
  """
  # Collect validated labels by bin.
  bin_pos = collections.defaultdict(int)
  bin_neg = collections.defaultdict(int)
  bin_weights = collections.defaultdict(float)
  for ex in examples:
    bin_weights[ex.bin] = ex.bin_weight
    if ex.is_pos == 1:
      bin_pos[ex.bin] += 1
    elif ex.is_pos == -1:
      bin_neg[ex.bin] += 1

  # P(+|b), P(-|b)
  p_pos_bin = {
      b: bin_pos[b] / (bin_pos[b] + bin_neg[b] + 1e-6) for b in bin_weights
  }
  p_neg_bin = {
      b: bin_neg[b] / (bin_pos[b] + bin_neg[b] + 1e-6) for b in bin_weights
  }
  # P(+), P(-) expected value.
  density_ev = np.array(
      [bin_weights[b] * p_pos_bin[b] for b in bin_weights]
  ).sum()
  p_bin_pos = collections.defaultdict(float)
  p_bin_neg = collections.defaultdict(float)
  for b in bin_weights:
    p_bin_pos[b] = bin_weights[b] * p_pos_bin[b] / density_ev
    p_bin_neg[b] = bin_weights[b] * p_neg_bin[b] / (1.0 - density_ev)

  roc_auc = 0

  # For off-diagonal bin pairs:
  # Take the probability of drawing a pos from bin j and neg from bin i.
  # If j > i, all pos examples are scored higher, so contributes directly to the
  # total ROC-AUC.
  bins = sorted(tuple(bin_weights.keys()))
  for i in range(len(bins)):
    for j in range(i + 1, len(bins)):
      roc_auc += p_bin_pos[j] * p_bin_neg[i]

  # For diagonal bin-pairs:
  # Look at actual in-bin observations for diagonal contribution.
  for b in bins:
    bin_pos_scores = np.array(
        [v.score for v in examples if v.bin == b and v.is_pos == 1]
    )
    bin_neg_scores = np.array(
        [v.score for v in examples if v.bin == b and v.is_pos == -1]
    )
    # If either is empty, there's no chance of pulling a (pos, neg) pair from
    # this bin, so we can continue.
    if bin_pos_scores.size == 0 or bin_neg_scores.size == 0:
      continue
    # Count the total number of (pos, neg) pairs where a pos example has a
    # higher score than a negative example.
    hits = (
        (bin_pos_scores[:, np.newaxis] - bin_neg_scores[np.newaxis, :]) > 0
    ).sum()
    bin_roc_auc = hits / (bin_pos_scores.size * bin_neg_scores.size)

    # Contribution is the probability of pulling both pos and neg examples
    # from this bin, multiplied by the bin's ROC-AUC.
    roc_auc += bin_roc_auc * p_bin_pos[b] * p_bin_neg[b]

  return roc_auc
