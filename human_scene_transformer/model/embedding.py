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

"""Contains embedding layers."""

import tensorflow as tf


class SinusoidalEmbeddingLayer(tf.keras.layers.Layer):
  """Sinusoidal Postional Embedding for xyz and time."""

  def __init__(self, min_freq=4, max_freq=256, hidden_size=256):
    super().__init__()
    self.min_freq = float(min_freq)
    self.max_freq = float(max_freq)
    self.hidden_size = hidden_size
    if hidden_size % 2 != 0:
      raise ValueError('hidden_size ({hidden_size}) must be divisible by 2.')
    self.num_freqs_int32 = hidden_size // 2
    self.num_freqs = tf.cast(self.num_freqs_int32, dtype=tf.float32)

  def build(self, input_shape):
    log_freq_increment = (
        tf.math.log(float(self.max_freq) / float(self.min_freq)) /
        tf.maximum(1.0, self.num_freqs - 1))
    # [num_freqs]
    self.inv_freqs = self.min_freq * tf.exp(
        tf.range(self.num_freqs, dtype=tf.float32) * -log_freq_increment)

  def call(self, input_tensor):
    # [..., num_freqs]
    input_tensor = tf.repeat(
        input_tensor[..., tf.newaxis], self.num_freqs_int32, axis=-1)
    # [..., h]
    embedded = tf.concat([
        tf.sin(input_tensor * self.inv_freqs),
        tf.cos(input_tensor * self.inv_freqs)
    ],
                         axis=-1)
    return embedded
