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

"""The ADE Keras Metric class."""

import gin
import tensorflow as tf


def distance_error(target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
  return tf.sqrt(
      tf.reduce_sum(tf.square(pred - target), axis=-1, keepdims=True))


@gin.configurable
class ADE(tf.keras.metrics.Metric):
  """Average Displacement Error over a n dimensional track.

  Calculates the mean L2 distance over all predicted timesteps.
  """

  def __init__(self, params, cutoff_seconds=None, at_cutoff=False, name='ADE'):
    """Initializes the ADE metric.

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

  def _reduce(self, ade_with_modes, input_batch, predictions):
    """Reduces mode dimension. The base class squeezes a single mode."""
    return tf.squeeze(ade_with_modes, axis=-1)

  def update_state(self, input_batch, predictions):
    should_predict = tf.cast(input_batch['should_predict'], tf.float32)

    target = input_batch[f'{self.agents_position_key}/target']
    target = target[..., :predictions['agents/position'].shape[-1]]
    # [b, a, t, n, 3] -> [b, a, t, n, 1]
    per_position_ade = distance_error(
        target[..., tf.newaxis, :],
        predictions['agents/position'])

    # Non-observed or past should not contribute to ade.
    deviation = tf.math.multiply_no_nan(per_position_ade,
                                        should_predict[..., tf.newaxis, :])
    # Chop off the un-wanted time part.
    # [b, a, cutoff_idx, 1]
    if self.at_cutoff and self.cutoff_seconds is not None:
      deviation = deviation[:, :, self.cutoff_idx-1:self.cutoff_idx, :]
      num_predictions = tf.reduce_sum(
          should_predict[:, :, self.cutoff_idx-1:self.cutoff_idx, :])
    else:
      deviation = deviation[:, :, :self.cutoff_idx, :]
      num_predictions = tf.reduce_sum(should_predict[:, :, :self.cutoff_idx, :])

    # Reduce along time
    deviation = tf.reduce_sum(deviation, axis=2)
    # Reduce along modes
    deviation = self._reduce(deviation, input_batch, predictions)
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
class MinADE(ADE):
  """Takes the minimum over all modes."""

  def _reduce(self, ade_with_modes, input_batch, predictions):
    return tf.reduce_min(ade_with_modes, axis=-2)


@gin.configurable
class MLADE(ADE):
  """Takes the maximum likelihood mode."""

  def _reduce(self, ade_with_modes, input_batch, predictions):
    # Get index of mixture component with highest probability
    # [b, a=1, t=1, n]
    ml_indices = tf.math.argmax(predictions['mixture_logits'], axis=-1)
    a = ade_with_modes.shape[1]
    ml_indices = tf.tile(
        tf.squeeze(ml_indices, axis=2), [1, a])[..., tf.newaxis]

    return tf.gather(
        ade_with_modes, indices=ml_indices, batch_dims=2, axis=-2)[..., 0, :]
