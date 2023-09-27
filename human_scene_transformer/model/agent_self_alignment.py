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

"""Contains agent self alignment layers."""

import tensorflow as tf


class AgentSelfAlignmentLayer(tf.keras.layers.Layer):
  """Enables agent to become aware of its temporal identity.

  Agent features are cross-attended with a learned query in temporal dimension.
  """

  def __init__(self,
               num_heads=8,
               hidden_size=256,
               ln_eps=1e-6,
               ff_dim=128,
               mask_style=None):
    super().__init__()
    self.hidden_size = hidden_size
    self.mask_style = mask_style
    if hidden_size % num_heads != 0:
      raise ValueError(f'hidden_size ({hidden_size}) must be an integer '
                       f'times bigger than num_heads ({num_heads}).')
    self.attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        attention_axes=2)
    self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f', output_shape=ff_dim, bias_axes='f', activation='relu')
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None)
    self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

    # [1, 1, 1, h]
    self.learned_query_vec = tf.Variable(
        tf.random_uniform_initializer(
            minval=-1., maxval=1.)(shape=[1, 1, 1, hidden_size]),
        trainable=True,
        dtype=tf.float32)

  def build_learned_query(self, input_batch):
    """Converts self.learned_query_vec into a learned query vector."""
    # This weird thing is for exporting and loading keras model...
    b = tf.shape(input_batch['hidden_vecs'])[0]
    a = tf.shape(input_batch['hidden_vecs'])[1]
    t = tf.shape(input_batch['hidden_vecs'])[2]

    # [b, a, t, 1, h]
    return tf.tile(self.learned_query_vec, [b, a, t, 1])

  def call(self, input_batch):
    input_batch = input_batch.copy()

    # [b, a, t, h]
    hidden_vecs = input_batch['hidden_vecs']

    # Expand the attention mask with new dims so that Keras can broadcast to
    # the same shape as the attn_score: [b, num_heads, a, t, a, t].
    # attn_mask shape: [b, 1, 1, 1, a, t,]
    # True means the position participate in the attention while all
    # False positions are ignored.
    if self.mask_style is None:
      attn_mask = None
    elif self.mask_style == 'has_historic_data':
      has_historic_data = tf.logical_and(
          input_batch['has_historic_data'][..., 0],
          tf.logical_not(input_batch['is_hidden'][..., 0]))
      attn_mask = has_historic_data[:, :, tf.newaxis, tf.newaxis, :]
    elif self.mask_style == 'has_data':
      has_data_historic = tf.logical_and(
          input_batch['has_data'][..., 0],
          tf.logical_not(input_batch['is_hidden'][..., 0]))
      attn_mask = has_data_historic[:, :, tf.newaxis, tf.newaxis, :,]
    else:
      raise ValueError(f'Unrecognized mask style: {self.mask_style}. '
                       "Must be either None, 'fully_padded' or 'any_padded'.")

    # [b, a, t, 1, h]
    learned_query = self.build_learned_query(input_batch)
    attn_out, attn_score = self.attn_layer(
        query=learned_query,
        key=hidden_vecs,
        value=hidden_vecs,
        attention_mask=attn_mask,
        return_attention_scores=True)

    attn_out = self.attn_ln(attn_out + hidden_vecs)

    # Feed-forward layers.
    out = self.ff_layer1(attn_out)
    out = self.ff_layer2(out)
    out = self.ff_ln(out + attn_out)

    input_batch['hidden_vecs'] = out
    input_batch[f'attn_scores/{self.name}'] = attn_score

    return input_batch
