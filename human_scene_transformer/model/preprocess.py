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

"""Contains preprocess layers."""

from typing import Dict, Optional

from human_scene_transformer.model.model_params import ModelParams

import tensorflow as tf


class PreprocessLayer(tf.keras.layers.Layer):
  """Preprocessing layer of the model.

  The preprocess layer applies the following preprocessing steps to the input.
  NOTE: we refer 'positions' as the transformer positions so each position
  corresponds to a unique agent at a specific timestep.

  1) Copy the raw input and save in `/original` and `/target` keys.
  2) Compute the 'has_data' mask for all features. True at fields where the
     feature has data. Further, compute the global 'has_data' mask. True where
     xyz data are available. Further, compute the 'has_historic_data' mask.
     True for agents which have at least one valid xyz data point in the xyz
     feature.
  3) Compute the `is_padded` bool tensor of shape [batch (b), num_agents (a)
     ,num_timesteps (t), 1]. True if the position is padded, ie, no valid
     observation.
  4) Compute which positions need to be predicted and save it to the
     `should_predict` bool tensor of shape [b, a, t, 1]. A position should be
     predicted if it is hidden, not padded and the agent has historic data.
  5) Mask agent features based on their 'has_data' mask.
  """

  def __init__(self, params: ModelParams):
    super().__init__(name='PreprocessLayer')
    self.params = params

    self.is_hidden_generator = params.is_hidden_generator(
        self.params.num_steps,
        self.params.num_history_steps)

    self.agents_feature_config = params.agents_feature_config
    self.agents_position_key = params.agents_position_key

  def call(self,
           raw_input_batch: Dict[str, tf.Tensor],
           is_hidden: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
    input_batch = raw_input_batch.copy()

    if is_hidden is None:
      num_agents = input_batch[self.params.agents_position_key].shape[1]
      input_batch['is_hidden'] = self.is_hidden_generator(
          num_agents, train_progress=0.0)
    else:
      input_batch['is_hidden'] = is_hidden

    input_batch = self._add_original_and_target(input_batch)
    input_batch = self._add_has_data(input_batch)
    input_batch = self._add_should_predict(input_batch)
    input_batch = self._mask_agent_features(input_batch)

    return input_batch

  def _add_original_and_target(self, input_batch):
    """Adds original, directly from input_batch values so we can use it later.
    """
    input_batch_new = input_batch.copy()
    for feature in self.agents_feature_config.keys():
      input_batch_new[f'{feature}/original'] = input_batch[feature]
      input_batch_new[f'{feature}/target'] = input_batch[feature]
      data_type = input_batch_new[f'{feature}/target'].dtype
      if data_type.is_floating:
        # Set target to 0. where NaN
        input_batch_new[f'{feature}/target'] = tf.where(
            tf.math.is_nan(input_batch_new[f'{feature}/target']),
            tf.zeros_like(input_batch_new[f'{feature}/target']),
            input_batch_new[f'{feature}/target'])
      else:
        input_batch_new[f'{feature}/target'] = tf.where(
            input_batch_new[f'{feature}/target'] == data_type.min,
            tf.zeros_like(input_batch_new[f'{feature}/target']),
            input_batch_new[f'{feature}/target'])
    return input_batch_new

  def _add_has_data(self, input_batch):
    num_hist_steps = self.params.num_history_steps
    input_batch = input_batch.copy()

    def has_data(t):
      if t.dtype.is_floating:
        return tf.math.logical_not(
            tf.reduce_any(tf.math.is_nan(t), axis=-1, keepdims=True))
      else:
        return tf.math.logical_not(
            tf.reduce_any(t == t.dtype.min, axis=-1, keepdims=True))

    # Each Feature
    for feature in self.agents_feature_config.keys():
      f = input_batch[feature]
      input_batch[f'has_data/{feature}'] = has_data(f)

    # Global
    # [b, a, t, 1]
    has_data = input_batch[f'has_data/{self.agents_position_key}']

    # [b, a, 1, 1]
    has_historic_data = tf.reduce_any(
        has_data[..., :num_hist_steps + 1, :], axis=-2, keepdims=True)

    input_batch['has_data'] = has_data
    input_batch['has_historic_data'] = has_historic_data
    return input_batch

  def _add_should_predict(self, input_batch):
    # Only include in loss computation if it is:
    # 1) hidden, 2) not padded, 3) has historic data
    input_batch['should_predict'] = tf.logical_and(
        input_batch['is_hidden'],
        tf.logical_and(input_batch['has_data'],
                       input_batch['has_historic_data']))

    return input_batch

  def _set_elems_to_value(self, target, should_set, new_val):
    """Sets elements in the target marked by should_set to value_to_set.

    Args:
      target: Target array to be operated on.
      should_set: This must be a binary array with value 1 or 0 with the same
        shape as the target. Elements with 1 will cause the element of the
        target at the same indices to be changed to value_to_set.
      new_val: The new value to set elements to.

    Returns:
      target: The target array after the operation.
    """

    target = tf.where(should_set, tf.cast(new_val, target.dtype), target)

    return target

  def _mask_agent_features(self, input_batch):
    input_batch = input_batch.copy()

    for feature in self.agents_feature_config.keys():
      feature_is_padded = tf.logical_not(input_batch[f'has_data/{feature}'])
      input_batch[feature] = self._set_elems_to_value(
          input_batch[feature],
          tf.logical_or(feature_is_padded, input_batch['is_hidden']), 0.)

    return input_batch
