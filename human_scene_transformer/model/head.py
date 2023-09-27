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

"""Contains prediction head layers."""

import gin

import tensorflow as tf


@gin.register
class PredictionHeadLayer(tf.keras.layers.Layer):
  """Converts transformer hidden vectors to model predictions."""

  def __init__(self, hidden_units=None):
    super().__init__()

    self.dense_layers = []
    # Add hidden layers.
    if hidden_units is not None:
      for units in hidden_units:
        self.dense_layers.append(
            tf.keras.layers.Dense(units, activation='relu'))
    self.dense_layers.append(
        tf.keras.layers.EinsumDense(
            '...h,hf->...f',
            output_shape=11,
            bias_axes='f',
            activation=None))

  def call(self, input_batch):
    # [b, a, t, h]
    hidden_vecs = input_batch['hidden_vecs']
    x = hidden_vecs
    # [b, a, t, 7]
    for layer in self.dense_layers:
      x = layer(x)
    pred = x
    predictions = {
        'agents/position': pred[..., 0:3],
        'agents/orientation': pred[..., 3:4],
        'agents/position/raw_scale_tril': pred[..., 4:10],
        'agents/orientation/raw_concentration': pred[..., 10:11],
    }
    if 'mixture_logits' in input_batch:
      predictions['mixture_logits'] = input_batch['mixture_logits']
    return predictions


@gin.register
class Prediction2DPositionHeadLayer(tf.keras.layers.Layer):
  """Converts transformer hidden vectors to model predictions."""

  def __init__(self, hidden_units=None, num_stages=5):
    super().__init__()

    self.dense_layers = []
    # Add hidden layers.
    if hidden_units is not None:
      for units in hidden_units:
        self.dense_layers.append(
            tf.keras.layers.Dense(units, activation='relu'))
    # Add the final prediction head.
    self.dense_layers.append(
        tf.keras.layers.EinsumDense(
            '...h,hf->...f',
            output_shape=5,
            bias_axes='f',
            activation=None))

  def call(self, input_batch):
    # [b, a, t, h]
    hidden_vecs = input_batch['hidden_vecs']
    x = hidden_vecs
    # [b, a, t, 5]
    for layer in self.dense_layers:
      x = layer(x)
    pred = x
    predictions = {
        'agents/position': pred[..., 0:2],
        'agents/position/raw_scale_tril': pred[..., 2:5],
    }
    if 'mixture_logits' in input_batch:
      predictions['mixture_logits'] = input_batch['mixture_logits']
    return predictions
