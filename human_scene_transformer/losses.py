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

"""Loss Functions."""

import gin

from human_scene_transformer.metrics import ade
from human_scene_transformer.model import output_distributions

import tensorflow as tf
import tensorflow_probability as tfp


class Loss(object):
  """Base class for Human Scene Transformer Losses.
  """

  def __init__(self, params, name='Loss', clip_loss_max=tf.float32.max):
    self.params = params
    self.clip_loss_max = clip_loss_max
    self.name = name

  @tf.function
  def __call__(self, input_batch, predictions):
    return self.call(input_batch, predictions)

  def call(self, input_batch, predictions):
    """Calculates loss for fields which should be predicted."""

    # [b, a, t]
    should_predict = input_batch['should_predict'][..., 0]
    # [b, a, t]
    loss_per_batch = self.get_per_batch_loss(input_batch, predictions)

    loss_per_batch = tfp.math.clip_by_value_preserve_gradient(
        loss_per_batch,
        tf.float32.min,
        self.clip_loss_max)

    # Compute loss only on positions w/ should_predict == True.
    should_predict_ind = tf.where(should_predict)
    loss_should_predict_mat = tf.gather_nd(
        params=loss_per_batch, indices=should_predict_ind)

    loss_should_predict = tf.reduce_mean(loss_should_predict_mat)
    # If there are no agents to be predicted xyz_loss_should_predict can be NaN
    loss_should_predict = tf.math.multiply_no_nan(
        loss_should_predict,
        tf.cast(tf.math.reduce_any(should_predict), tf.float32))

    loss_dict = {
        'loss': loss_should_predict,
        f'{self.name}_loss': loss_should_predict
    }
    return loss_dict

  def get_per_batch_loss(self, input_batch, predictions):
    raise NotImplementedError


@gin.register
class PositionOrientationNLLLoss(Loss):
  """Position and Orientation Loss for human trajectory predictions."""

  def __init__(self, params, **kwargs):
    super().__init__(params, name='position_orientation', **kwargs)
    self.position_loss_obj = PositionNLLLoss(params, **kwargs)
    self.orientation_loss_obj = OrientationNLLLoss(params, **kwargs)

  def call(self, input_batch, predictions):
    position_loss = self.position_loss_obj(input_batch, predictions)
    orientation_loss = self.orientation_loss_obj(input_batch, predictions)

    loss = position_loss['loss'] + orientation_loss['loss']

    loss_dict = position_loss | orientation_loss

    loss_dict['loss'] = loss

    return loss_dict


@gin.register
class MultimodalPositionOrientationNLLLoss(Loss):
  """Position and Orientation Loss for human trajectory predictions."""

  def __init__(self, params, **kwargs):
    super().__init__(params)
    self.position_loss_obj = MultimodalPositionNLLLoss(params, **kwargs)
    self.orientation_loss_obj = MultimodalOrientationNLLLoss(params, **kwargs)

  def call(self, input_batch, predictions):
    position_loss = self.position_loss_obj(input_batch, predictions)
    orientation_loss = self.orientation_loss_obj(input_batch, predictions)

    loss = position_loss['loss'] + orientation_loss['loss']

    loss_dict = position_loss | orientation_loss

    loss_dict['loss'] = loss

    return loss_dict


@gin.register
class OrientationNLLLoss(Loss):
  """Orientation NLL Loss for human trajectory predictions."""

  def __init__(self, params, **kwargs):
    super().__init__(params, name='orientation', **kwargs)

  def get_per_batch_loss(self, input_batch, predictions):
    """Negative log probability of ground truth with respect to predictions."""

    p_orientation = output_distributions.get_orientation_distribution(
        predictions)

    # [b, a, t, 1]
    orientation_nll = -p_orientation.log_prob(
        input_batch[f'{self.params.agents_orientation_key}/target'][..., 0])

    return orientation_nll


@gin.register
class MultimodalOrientationNLLLoss(Loss):
  """Orientation Loss for human trajectory predictions w/ scene transformer."""

  def __init__(self, params, **kwargs):
    super().__init__(params, name='orientation', **kwargs)

  def get_per_batch_loss(self, input_batch, predictions):
    """Negative log probability of ground truth with respect to predictions."""

    p_orientation_mm = (output_distributions
                        ).get_multimodal_orientation_distribution(predictions)

    # [b, a, t, 1]
    orientation_nll = -p_orientation_mm.log_prob(
        input_batch[f'{self.params.agents_orientation_key}/target'][..., 0])

    return orientation_nll


@gin.register
class PositionMSELoss(Loss):
  """Position MSE Loss for human trajectory predictions."""

  def __init__(self, params, **kwargs):
    super().__init__(params, name='position', **kwargs)

  def get_per_batch_loss(self, input_batch, predictions):
    # position
    # [b, a, t, 3]
    diff = tf.square((predictions['agents/position'] -
                      input_batch[f'{self.params.agents_position_key}/target']))
    # [b, a, t]
    diff = tf.reduce_sum(diff, axis=-1)

    diff = tf.sqrt(diff)

    return diff


@gin.register
class PositionNLLLoss(Loss):
  """Position NLL Loss for human trajectory predictions."""

  def __init__(self, params, **kwargs):
    super().__init__(params, name='position', **kwargs)

  def get_per_batch_loss(self, input_batch, predictions):
    """Negative log probability of ground truth with respect to predictions."""
    # [b, a, t, 3]
    p_position = output_distributions.get_position_distribution(predictions)

    # [b, a, t, 1]
    position_nll = -p_position.log_prob(
        input_batch[f'{self.params.agents_position_key}/target'])
    return position_nll


@gin.register
class MultimodalPositionNLLLoss(Loss):
  """Position loss for human trajectory predictions w/ scene transformer."""

  def __init__(self, params, **kwargs):
    super().__init__(params, name='position', **kwargs)

  def get_per_batch_loss(self, input_batch, predictions):
    """Negative log probability of ground truth with respect to predictions."""
    # [b, a, t, n, 3]
    p_position_mm = output_distributions.get_multimodal_position_distribution(
        predictions)

    target = input_batch[f'{self.params.agents_position_key}/target']
    target = target[..., :p_position_mm.event_shape_tensor()[0]]

    # [b, a, t]
    position_nll = -p_position_mm.log_prob(target)

    return position_nll


@gin.register
class MinAdePositionNLLMixtureCategoricalCrossentropyLoss(Loss):
  """MinADEPositionNLLLoss and MixtureCategoricalCrossentropyLoss."""

  def __init__(self, params, **kwargs):
    super().__init__(params, name='MinAdeNLLMixture', **kwargs)
    self.position_loss_obj = MinADEPositionNLLLoss(params)
    self.mixture_loss_obj = MinAdeMixtureCategoricalCrossentropyLoss(params)

  def call(self, input_batch, predictions):
    position_loss = self.position_loss_obj(input_batch, predictions)
    mixture_loss = self.mixture_loss_obj(input_batch, predictions)

    loss = position_loss['loss'] + mixture_loss['loss']

    loss_dict = position_loss | mixture_loss

    loss_dict['loss'] = loss

    return loss_dict


@gin.register
class MinADEPositionNLLLoss(PositionNLLLoss):
  """MinAdePositionNLL loss."""

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    self.mixture_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  def get_per_batch_loss(self, input_batch, predictions):
    """Negative log probability of mode with smallest ADE."""

    input_batch = input_batch.copy()
    input_batch[f'{self.params.agents_position_key}/target'] = input_batch[
        f'{self.params.agents_position_key}/target'][..., tf.newaxis, :]
    # [b, a, t, n]
    position_nll = super().get_per_batch_loss(input_batch, predictions)

    # Calculate ADE
    # [b, a, t, n, 1]
    per_position_ade = ade.distance_error(
        input_batch[f'{self.params.agents_position_key}/target'],
        predictions[f'{self.params.agents_position_key}'])

    # [b, a, t, 1]
    should_predict = tf.cast(input_batch['should_predict'], tf.float32)

    # [b, a, t, n, 1]
    per_position_ade = per_position_ade * should_predict[..., tf.newaxis, :]

    # Get mode with minimum ADE
    # [b, a, n, 1]
    per_mode_ade_sum = tf.reduce_sum(per_position_ade, axis=2)

    t = tf.shape(position_nll)[2]

    # [b, a, 1]
    min_ade_indices = tf.math.argmin(per_mode_ade_sum, axis=-2)

    # [b, a, t, 1]
    min_ade_indices_tiled = tf.tile(
        min_ade_indices[..., tf.newaxis], [1, 1, t, 1])

    # [b, a, t]
    position_nll_min_ade = tf.gather(
        position_nll, indices=min_ade_indices_tiled, batch_dims=3, axis=-1
        )[..., 0]

    return position_nll_min_ade


@gin.register
class MinAdeMixtureCategoricalCrossentropyLoss(Loss):
  """Categorical Corssentropy Loss for Mixture Weight."""

  def __init__(self, params, **kwargs):
    super().__init__(params, name='mixture', **kwargs)
    self.mixture_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  def get_per_batch_loss(self, input_batch, predictions):
    """Negative log probability of ground truth with respect to predictions."""
    input_batch = input_batch.copy()
    input_batch[f'{self.params.agents_position_key}/target'] = input_batch[
        f'{self.params.agents_position_key}/target'][..., tf.newaxis, :]

    # Calculate ADE
    # [b, a, t, n, 1]
    per_position_ade = ade.distance_error(
        input_batch[f'{self.params.agents_position_key}/target'],
        predictions[f'{self.params.agents_position_key}'])

    # [b, a, t, 1]
    should_predict = tf.cast(input_batch['should_predict'], tf.float32)

    # [b, a, t, n, 1]
    per_position_ade = per_position_ade * should_predict[..., tf.newaxis, :]

    # Get mode with minimum ADE
    # [b, a, n, 1]
    per_mode_ade_sum = tf.reduce_sum(per_position_ade, axis=2)

    a = tf.shape(per_position_ade)[1]
    n = tf.shape(per_position_ade)[3]

    # [b, a, 1]
    min_ade_indices = tf.math.argmin(per_mode_ade_sum, axis=-2)

    # [b, a, n]
    min_ade_indices_one_hot = tf.one_hot(min_ade_indices[..., 0], n)

    # [b, a]
    mixture_loss = self.mixture_loss(
        min_ade_indices_one_hot,
        tf.tile(predictions['mixture_logits'][..., 0, :], [1, a, 1]))

    return mixture_loss

  def call(self, input_batch, predictions):
    """Calculates loss."""

    # [b, a]
    should_predict = tf.reduce_any(
        input_batch['should_predict'][..., 0], axis=-1)
    # [b, a]
    loss_per_batch = self.get_per_batch_loss(input_batch, predictions)

    # Compute loss only on positions w/ should_predict == True.
    should_predict_ind = tf.where(should_predict)
    loss_should_predict_mat = tf.gather_nd(
        params=loss_per_batch, indices=should_predict_ind)

    loss_should_predict = tf.reduce_mean(loss_should_predict_mat)
    # If there are no agents to be predicted xyz_loss_should_predict can be NaN
    loss_should_predict = tf.math.multiply_no_nan(
        loss_should_predict,
        tf.cast(tf.math.reduce_any(should_predict), tf.float32))

    # Mixture weights are per scene. So we do not have to mask anything
    loss_dict = {
        'loss': loss_should_predict,
        f'{self.name}_loss': loss_should_predict
    }
    return loss_dict


@gin.register
class MinAdePositionMixtureCategoricalCrossentropyLoss(Loss):
  """MinADEPositionNLLLoss and MixtureCategoricalCrossentropyLoss."""

  def __init__(self, params, **kwargs):
    super().__init__(params, name='MinAdeNLLMixture', **kwargs)
    self.position_loss_obj = MinADEPositionLoss(params)
    self.mixture_loss_obj = MinAdeMixtureCategoricalCrossentropyLoss(params)

  def call(self, input_batch, predictions):
    position_loss = self.position_loss_obj(input_batch, predictions)
    mixture_loss = self.mixture_loss_obj(input_batch, predictions)

    loss = position_loss['loss'] + mixture_loss['loss']

    loss_dict = position_loss | mixture_loss

    loss_dict['loss'] = loss

    return loss_dict


@gin.register
class MinADEPositionLoss(PositionNLLLoss):
  """MinAdePositionNLL loss."""

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    self.mixture_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  def get_per_batch_loss(self, input_batch, predictions):
    """Negative log probability of mode with smallest ADE."""

    input_batch = input_batch.copy()
    input_batch[f'{self.params.agents_position_key}/target'] = input_batch[
        f'{self.params.agents_position_key}/target'][..., tf.newaxis, :]

    # Calculate ADE
    # [b, a, t, n, 1]
    per_position_ade = ade.distance_error(
        input_batch[f'{self.params.agents_position_key}/target'],
        predictions[f'{self.params.agents_position_key}'])

    per_position_ade_org = per_position_ade

    # [b, a, t, 1]
    should_predict = tf.cast(input_batch['should_predict'], tf.float32)

    # [b, a, t, n, 1]
    per_position_ade = per_position_ade * should_predict[..., tf.newaxis, :]

    # Get mode with minimum ADE
    # [b, a, n, 1]
    per_mode_ade_sum = tf.reduce_sum(per_position_ade, axis=2)

    t = tf.shape(per_position_ade)[2]

    # [b, a, 1]

    # [b, a, t, 1]
    min_ade_indices_tiled = tf.tile(
        per_mode_ade_sum[..., tf.newaxis], [1, 1, t, 1]
    )

    # [b, a, t]
    position_min_ade = tf.gather(
        per_position_ade_org[..., 0],
        indices=min_ade_indices_tiled,
        batch_dims=3,
        axis=-1,
    )[..., 0]

    return position_min_ade


@gin.register
class MinNLLPositionMixtureCategoricalCrossentropyLoss(Loss):
  """MinNLLPositionNLLLoss and MixtureCategoricalCrossentropyLoss."""

  def __init__(self, params, **kwargs):
    super().__init__(params, name='MinNLLMixture', **kwargs)
    self.position_loss_obj = MinNLLPositionLoss(params)
    self.mixture_loss_obj = MinNLLMixtureCategoricalCrossentropyLoss(params)

  def call(self, input_batch, predictions):
    position_loss = self.position_loss_obj(input_batch, predictions)
    mixture_loss = self.mixture_loss_obj(input_batch, predictions)

    loss = position_loss['loss'] + mixture_loss['loss']

    loss_dict = position_loss | mixture_loss

    loss_dict['loss'] = loss

    return loss_dict


@gin.register
class MinNLLPositionLoss(PositionNLLLoss):
  """MinNLLPositionNLL loss."""

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    self.mixture_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  def get_per_batch_loss(self, input_batch, predictions):
    """Negative log probability of mode with smallest ADE."""

    input_batch = input_batch.copy()
    input_batch[f'{self.params.agents_position_key}/target'] = input_batch[
        f'{self.params.agents_position_key}/target'][..., tf.newaxis, :]
    # [b, a, t, n]
    position_nll = super().get_per_batch_loss(input_batch, predictions)

    # [b, a, t, 1]
    should_predict = tf.cast(input_batch['should_predict'], tf.float32)

    # [b, a, t, n, 1]
    per_position_nll = (
        position_nll[..., tf.newaxis] * should_predict[..., tf.newaxis, :]
    )

    # Get mode with minimum NLL
    # [b, a, n, 1]
    per_mode_nll_sum = tf.reduce_sum(per_position_nll, axis=2)

    t = tf.shape(position_nll)[2]

    # [b, a, 1]
    min_nll_indices = tf.math.argmin(per_mode_nll_sum, axis=-2)

    # [b, a, t, 1]
    min_nll_indices_tiled = tf.tile(
        min_nll_indices[..., tf.newaxis], [1, 1, t, 1])

    # [b, a, t]
    position_nll_min_ade = tf.gather(
        position_nll, indices=min_nll_indices_tiled, batch_dims=3, axis=-1
        )[..., 0]

    return position_nll_min_ade


@gin.register
class MinNLLMixtureCategoricalCrossentropyLoss(PositionNLLLoss):
  """Categorical Corssentropy Loss for Mixture Weight."""

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    self.mixture_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  def get_per_batch_loss(self, input_batch, predictions):
    """Negative log probability of ground truth with respect to predictions."""
    input_batch = input_batch.copy()
    input_batch[f'{self.params.agents_position_key}/target'] = input_batch[
        f'{self.params.agents_position_key}/target'][..., tf.newaxis, :]

    # Calculate ADE
    # [b, a, t, n, 1]
    position_nll = super().get_per_batch_loss(input_batch, predictions)

    # [b, a, t, 1]
    should_predict = tf.cast(input_batch['should_predict'], tf.float32)

    # [b, a, t, n, 1]
    per_position_nll = (
        position_nll[..., tf.newaxis] * should_predict[..., tf.newaxis, :]
    )

    # Get mode with minimum NLL
    # [b, a, n, 1]
    per_mode_nll_sum = tf.reduce_sum(per_position_nll, axis=2)

    a = tf.shape(position_nll)[1]
    n = tf.shape(position_nll)[3]

    # [b, a, 1]
    min_nll_indices = tf.math.argmin(per_mode_nll_sum, axis=-2)

    # [b, a, n]
    min_nll_indices_one_hot = tf.one_hot(min_nll_indices[..., 0], n)

    # [b, a]
    mixture_loss = self.mixture_loss(
        min_nll_indices_one_hot,
        tf.tile(predictions['mixture_logits'][..., 0, :], [1, a, 1]))

    return mixture_loss

  def call(self, input_batch, predictions):
    """Calculates loss."""

    # [b, a]
    should_predict = tf.reduce_any(
        input_batch['should_predict'][..., 0], axis=-1)
    # [b, a]
    loss_per_batch = self.get_per_batch_loss(input_batch, predictions)

    # Compute loss only on positions w/ should_predict == True.
    should_predict_ind = tf.where(should_predict)
    loss_should_predict_mat = tf.gather_nd(
        params=loss_per_batch, indices=should_predict_ind)

    loss_should_predict = tf.reduce_mean(loss_should_predict_mat)
    # If there are no agents to be predicted xyz_loss_should_predict can be NaN
    loss_should_predict = tf.math.multiply_no_nan(
        loss_should_predict,
        tf.cast(tf.math.reduce_any(should_predict), tf.float32))

    # Mixture weights are per scene. So we do not have to mask anything
    loss_dict = {
        'loss': loss_should_predict,
        f'{self.name}_loss': loss_should_predict
    }
    return loss_dict
