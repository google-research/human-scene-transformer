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

"""Contains scene encoder layers."""

import gin

from human_scene_transformer.model import embedding

import tensorflow as tf


@gin.register
class ConvOccupancyGridEncoderLayer(tf.keras.layers.Layer):
  """Uses only the frame on the current step for now."""

  def __init__(self, params):
    super().__init__()
    self.xyz_embedding_layer = embedding.SinusoidalEmbeddingLayer(
        hidden_size=params.hidden_size)
    self.current_step_idx = params.num_history_steps
    self.num_filters = params.num_conv_filters
    self.hidden_size = params.hidden_size
    drop_prob = params.drop_prob
    self.mlp = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=self.xyz_embedding_layer.hidden_size,
        bias_axes='h',
        activation=None)

    # Construct conv and max pooling layers.
    self.layers = []
    for i, num_filter in enumerate(self.num_filters):
      if i == 0 or i == 1:
        strides = 2
      else:
        strides = 1
      conv_layer = tf.keras.layers.Conv2D(
          filters=num_filter,
          kernel_size=3,
          strides=strides,
          padding='same',
          activation='relu',
          data_format='channels_last')
      self.layers.append(conv_layer)
      dropout = tf.keras.layers.Dropout(drop_prob)
      self.layers.append(dropout)
      pooling_layer = tf.keras.layers.MaxPool2D(
          pool_size=(2, 2), data_format='channels_last')
      self.layers.append(pooling_layer)
    # Flatten and use MLP to map to [b, h].
    self.layers.append(tf.keras.layers.Flatten())
    self.layers.append(
        tf.keras.layers.Dense(units=self.hidden_size, activation='relu'))
    self.layers.append(tf.keras.layers.Dropout(drop_prob))

    self.seq_layers = tf.keras.Sequential(self.layers)

  def _get_origin_embedding(self, input_batch, training=None):
    # [b]
    x = input_batch['scene/origin'][:, self.current_step_idx, 0]
    y = input_batch['scene/origin'][:, self.current_step_idx, 1]
    # [b, hidden_size]
    origin_embedding_x = self.xyz_embedding_layer(x, training=training)
    origin_embedding_y = self.xyz_embedding_layer(y, training=training)
    origin_embedding = tf.concat([origin_embedding_x, origin_embedding_y],
                                 axis=-1)
    origin_embedding = self.mlp(origin_embedding)

    return origin_embedding

  def _apply_img_layers(self, input_batch, training=None):
    # [b, width, height].
    occ_grid = input_batch['scene/grid'][:, self.current_step_idx, :, :,
                                         tf.newaxis]

    occ_grid = self.seq_layers(occ_grid, training=training)
    return occ_grid

  def call(self, input_batch, training=None):
    input_batch = input_batch.copy()

    # Embed occupation grid origin w/ sinusoidal embedding.
    origin_embedding = self._get_origin_embedding(input_batch)

    # Apply layers.
    occ_grid = self._apply_img_layers(input_batch, training=training)

    # [b, hidden_size]. Combine origin and grid.
    occ_grid += origin_embedding
    input_batch['scene_hidden_vec'] = occ_grid[:, tf.newaxis, tf.newaxis :]
    return input_batch


@gin.register
class PointCloudEncoderLayer(tf.keras.layers.Layer):
  """Retrieves the point cloud at the current timestep."""

  def __init__(self, params):
    super().__init__()
    self.current_step_idx = params.num_history_steps
    self.embedding_layer = embedding.SinusoidalEmbeddingLayer(
        hidden_size=params.feature_embedding_size)

  def call(self, input_batch, training=None):
    input_batch = input_batch.copy()

    pc = input_batch['scene/pc'][:, self.current_step_idx, ..., :2]

    pc = tf.where(tf.math.is_nan(pc), 0., pc)

    pc_emb = self.embedding_layer(pc, training=training)

    pc_emb = tf.concat([pc_emb[..., 0, :], pc_emb[..., 1, :]], axis=-1)

    input_batch['scene_hidden_vec'] = pc_emb

    return input_batch
