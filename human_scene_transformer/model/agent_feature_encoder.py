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

"""Agent feature encoder."""

import gin

from human_scene_transformer.model.embedding import SinusoidalEmbeddingLayer

import tensorflow as tf


@gin.register
class AgentTemporalEncoder(tf.keras.layers.Layer):
  """Encodes agents temporal positions."""

  def __init__(self, key, output_shape, params):
    super().__init__()
    self.key = key
    self.embedding_layer = SinusoidalEmbeddingLayer(
        max_freq=params.num_steps,
        hidden_size=params.feature_embedding_size)

    self.mlp = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        activation=None)

  def _get_temporal_embedding(self, input_batch):
    # This weird thing is for exporting and loading keras model...
    b = tf.shape(input_batch[self.key])[0]
    num_agents = tf.shape(input_batch[self.key])[1]
    num_steps = tf.shape(input_batch[self.key])[2]

    t = tf.range(0, num_steps, dtype=tf.float32)
    t = t[tf.newaxis, tf.newaxis, :]
    t = tf.tile(t, [b, num_agents, 1])
    return self.embedding_layer(t[..., tf.newaxis])

  def call(self, input_batch):
    return (self.mlp(self._get_temporal_embedding(input_batch)),
            tf.ones_like(input_batch['has_data']))


@gin.register
class AgentPositionEncoder(tf.keras.layers.Layer):
  """Encodes agents spatial positions."""

  def __init__(self, key, output_shape, params):
    super().__init__()
    self.key = key
    self.embedding_layer = SinusoidalEmbeddingLayer(
        hidden_size=params.feature_embedding_size)

    self.mlp = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        activation=None)

  def call(self, input_batch):
    not_is_hidden = tf.logical_not(input_batch['is_hidden'])
    mask = tf.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
    mask = tf.repeat(mask, tf.shape(input_batch[self.key])[-1], axis=-1)
    return self.mlp(self.embedding_layer(input_batch[self.key])), mask


@gin.register
class Agent2DOrientationEncoder(tf.keras.layers.Layer):
  """Encodes agents 2d orientation."""

  def __init__(self, key, output_shape, params):
    super().__init__()
    self.key = key
    self.embedding_layer = SinusoidalEmbeddingLayer(
        max_freq=2,
        hidden_size=params.feature_embedding_size//2)

    self.mlp = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        activation=None)

  def call(self, input_batch):
    orientation = input_batch[self.key]
    orientation_embedding = tf.concat([
        self.embedding_layer(tf.math.sin(orientation)),
        self.embedding_layer(tf.math.cos(orientation))
    ], axis=-1)

    not_is_hidden = tf.logical_not(input_batch['is_hidden'])
    mask = tf.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)

    return self.mlp(orientation_embedding), mask


@gin.register
class AgentScalarEncoder(tf.keras.layers.Layer):
  """Encodes a agent's scalar."""

  def __init__(self, key, output_shape, params):
    super().__init__()
    self.key = key

    self.mlp = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        activation=None)

  def call(self, input_batch):
    not_is_hidden = tf.logical_not(input_batch['is_hidden'])
    mask = tf.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
    return self.mlp(input_batch[self.key])[..., tf.newaxis, :], mask


@gin.configurable
class AgentOneHotEncoder(tf.keras.layers.Layer):
  """Encodes the detection stage."""

  def __init__(self, key, output_shape, params, depth=1):
    super().__init__()
    self.key = key
    self.depth = depth

    self.mlp = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        activation=None)

  def call(self, input_batch):
    stage_one_hot = tf.one_hot(
        tf.squeeze(input_batch[self.key], axis=-1),
        self.depth)

    not_is_hidden = tf.logical_not(input_batch['is_hidden'])
    mask = tf.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
    return self.mlp(stage_one_hot)[..., tf.newaxis, :], mask


@gin.configurable
class AgentKeypointsEncoder(tf.keras.layers.Layer):
  """Encodes the agent's keypoints."""

  def __init__(self, key, output_shape, params):
    super().__init__()
    self.key = key

    self.mlp1 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        activation=tf.nn.relu)

  def call(self, input_batch, training=None):
    not_is_hidden = tf.logical_not(input_batch['is_hidden'])
    mask = tf.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)

    keypoints = input_batch[self.key]

    out = self.mlp1(keypoints)[..., tf.newaxis, :]

    return out, mask


@gin.configurable
class AgentHeadOrientationEncoder(tf.keras.layers.Layer):
  """Encodes the detection stage."""

  def __init__(self, key, output_shape, params):
    super().__init__()
    self.key = key

    self.mlp = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        activation=None)

  def call(self, input_batch):
    not_is_hidden = tf.logical_not(input_batch['is_hidden'])
    mask = tf.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
    return self.mlp(input_batch[self.key])[..., tf.newaxis, :], mask
