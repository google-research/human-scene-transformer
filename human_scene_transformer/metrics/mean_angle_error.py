# Copyright 2023 The human_scene_transformer Authors.
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

"""The Mean Angle Error Keras Metric class."""

import gin
import numpy as np
import tensorflow as tf


@gin.configurable
class MeanAngleError(tf.keras.metrics.Metric):
  """Mean Angle Error over yaw angle.

  Calculates the mean angular distance over all predicted timesteps.
  """

  def __init__(self, params, cutoff_seconds=None, at_cutoff=False,
               name='AngleError'):
    """Initializes the MeanAngleError metric.

    Args:
      params: ModelParams
      cutoff_seconds: Cutoff up to which time the metric should be calculated
        in seconds.
      at_cutoff: If True metric will be calculated at cutoff timestep.
        Otherwise metric is calculated as average up to cutoff_seconds.
      name: Metric name.
    """
    super().__init__(name=name)
    self.agents_orientation_key = params.agents_orientation_key
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

  def _reduce(self, mae, input_batch, predictions):
    return mae[..., 0, :]

  def update_state(self, input_batch, predictions):
    should_predict = tf.cast(input_batch['should_predict'], tf.float32)
    # [b, a, t, n, 1]
    diff = (
        input_batch[f'{self.agents_orientation_key}/target'][..., tf.newaxis, :]
        - predictions['agents/orientation'])
    deviation = tf.abs(tf.math.mod((diff + np.pi), (2 * np.pi)) - np.pi)

    deviation = self._reduce(deviation, input_batch, predictions)

    # Non-observed or past should not contribute to ade.
    deviation = tf.math.multiply_no_nan(deviation, should_predict)
    # Chop off the un-wanted time part.
    # [b, a, cutoff_idx, 1]
    if self.at_cutoff and self.cutoff_seconds is not None:
      deviation = deviation[:, :, self.cutoff_idx-1:self.cutoff_idx, :]
      num_predictions = tf.reduce_sum(
          should_predict[:, :, self.cutoff_idx-1::self.cutoff_idx, :])
    else:
      deviation = deviation[:, :, :self.cutoff_idx, :]
      num_predictions = tf.reduce_sum(should_predict[:, :, :self.cutoff_idx, :])
    # [1]
    deviation = tf.reduce_sum(deviation)

    self.num_predictions.assign_add(num_predictions)
    self.total_deviation.assign_add(deviation)

  def result(self):
    return self.total_deviation / self.num_predictions

  def reset_states(self):
    self.num_predictions.assign(0)
    self.total_deviation.assign(0.0)


@gin.configurable
class MinMeanAngleError(MeanAngleError):
  """Takes the minimum over all modes."""

  def _reduce(self, mae, input_batch, predictions):
    return tf.reduce_min(mae, axis=-2)


@gin.configurable
class MLMeanAngleError(MeanAngleError):
  """Takes the maximum likelihood mode."""

  def _reduce(self, mae, input_batch, predictions):
    # Get index of mixture component with highest probability
    ml_indices = tf.math.argmax(predictions['mixture_logits'], axis=-1)
    a = mae.shape[1]
    t = mae.shape[2]
    ml_indices = tf.tile(ml_indices, [1, a, t])[..., tf.newaxis]

    return tf.gather(mae, indices=ml_indices, batch_dims=3, axis=-2)[..., 0, :]

