# Copyright 2024 The human_scene_transformer Authors.
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

"""Position Negative Log Likelihood Keras metric."""

import gin

from human_scene_transformer.model import output_distributions

import tensorflow as tf


@gin.configurable
class PositionNegativeLogLikelihood(tf.keras.metrics.Metric):
  """Position Negative Log Likelihood."""

  def __init__(self, params, cutoff_seconds=None, at_cutoff=False,
               name='PosNLL'):
    """Initializes the PositionNegativeLogLikelihood metric.

    Args:
      params: ModelParams
      cutoff_seconds: Cutoff up to which time the metric should be calculated
        in seconds.
      at_cutoff: If True metric will be calculated at cutoff timestep.
        Otherwise metric is calculated as average up to cutoff_seconds.
      name: Metric name.
    """
    super().__init__(name=name)
    self.agents_position_key = params.agents_position_key
    self.cutoff_seconds = cutoff_seconds
    self.at_cutoff = at_cutoff
    if cutoff_seconds is None:
      self.cutoff_idx = None
    else:
      # +1 due to current time step.
      self.cutoff_idx = int(
          params.num_history_steps +
          cutoff_seconds / params.timestep) + 1

    self.num_predictions = self.add_weight(
        name='num_predictions', initializer='zeros')
    self.total_deviation = self.add_weight(
        name='total_deviation', initializer='zeros')

  def update_state(self, input_batch, predictions):
    should_predict = tf.cast(input_batch['should_predict'], tf.float32)

    p_pos = output_distributions.get_multimodal_position_distribution(
        predictions)

    target = input_batch[f'{self.agents_position_key}/target']
    target = target[..., :p_pos.event_shape_tensor()[0]]

    # [b, a, t, n, 1]
    per_position_nll = -p_pos.log_prob(target)[..., tf.newaxis]

    # Non-observed or past should not contribute to metric.
    nll = tf.math.multiply_no_nan(per_position_nll, should_predict)
    # Chop off the un-wanted time part.
    # [b, a, cutoff_idx, 1]
    if self.at_cutoff and self.cutoff_seconds is not None:
      nll = nll[:, :, self.cutoff_idx-1:self.cutoff_idx, :]
      num_predictions = tf.reduce_sum(
          should_predict[:, :, self.cutoff_idx-1::self.cutoff_idx, :])
    else:
      nll = nll[:, :, :self.cutoff_idx, :]
      num_predictions = tf.reduce_sum(should_predict[:, :, :self.cutoff_idx, :])

    # [1]
    nll = tf.reduce_sum(nll)

    self.num_predictions.assign_add(num_predictions)
    self.total_deviation.assign_add(nll)

  def result(self):
    return self.total_deviation / self.num_predictions

  def reset_states(self):
    self.num_predictions.assign(0)
    self.total_deviation.assign(0.0)
