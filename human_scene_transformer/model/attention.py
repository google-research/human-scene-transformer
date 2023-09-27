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

"""Contains attention layers."""

import tensorflow as tf


class SelfAttnTransformerLayer(tf.keras.layers.Layer):
  """Performs full self-attention across the agent and time dimensions."""

  def __init__(
      self,
      num_heads=8,
      hidden_size=256,
      drop_prob=0.1,
      ln_eps=1e-6,
      ff_dim=128,
      mask_style=None,
      flatten=False,
      multimodality_induced=False,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.mask_style = mask_style
    self.flatten = flatten
    self.multimodality_induced = multimodality_induced
    if hidden_size % num_heads != 0:
      raise ValueError(
          f'hidden_size ({hidden_size}) must be an integer '
          f'times bigger than num_heads ({num_heads}).'
      )
    self.attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        attention_axes=1 if flatten else (1, 2),
    )  # Full Attention over agents and time
    self.attn_dropout = tf.keras.layers.Dropout(drop_prob)
    self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f', output_shape=ff_dim, bias_axes='f', activation='relu'
    )
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None,
    )
    self.ff_dropout = tf.keras.layers.Dropout(drop_prob)
    self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

  def call(self, input_batch, training=None):
    input_batch = input_batch.copy()

    # [b, a, t, h] or [b, a, t, n, h]
    hidden_vecs = input_batch['hidden_vecs']

    if self.flatten:
      h_shape = tf.shape(input_batch['hidden_vecs'])
      b = h_shape[0]
      h = h_shape[-1]

      if self.multimodality_induced:
        n = h_shape[3]
        hidden_vecs = tf.reshape(hidden_vecs, (b, -1, n, h))
      else:
        hidden_vecs = tf.reshape(hidden_vecs, (b, -1, h))

    # Expand the attention mask with new dims so that Keras can broadcast to
    # the same shape as the attn_score: [b, num_heads, a, t, a, t].
    # attn_mask shape: [b, 1, 1, 1, a, t,]
    # True means the position participate in the attention while all
    # False positions are ignored.
    if self.mask_style is None:
      attn_mask = None
    elif self.mask_style == 'has_historic_data':
      has_historic_data = input_batch['has_historic_data'][..., 0]
      attn_mask = has_historic_data[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    else:
      raise ValueError(f'Unrecognized mask style: {self.mask_style}. '
                       "Must be either None, 'fully_padded' or 'any_padded'.")

    if attn_mask is not None and self.flatten:
      a = h_shape[1]
      t = h_shape[2]

      if self.multimodality_induced:  # We have modes
        n = h_shape[3]
        attn_mask_with_modes = attn_mask[..., tf.newaxis, :, :, :, :]
        tiled_mask = tf.tile(attn_mask_with_modes, [1, 1, 1, a, t, 1, t])
        attn_mask = tf.reshape(
            tiled_mask,
            [b, 1, 1, tf.cast(a*t, tf.int32), tf.cast(a*t, tf.int32)]
        )
      else:
        tiled_mask = tf.tile(attn_mask, [1, 1, a, t, 1, t])
        attn_mask = tf.reshape(
            tiled_mask,
            [b, 1, tf.cast(a*t, tf.int32), tf.cast(a*t, tf.int32)]
        )

    attn_out, attn_score = self.attn_layer(
        query=hidden_vecs,
        key=hidden_vecs,
        value=hidden_vecs,
        attention_mask=attn_mask,
        return_attention_scores=True)
    out = self.attn_dropout(attn_out, training=training)
    attn_out = self.attn_ln(out + hidden_vecs)

    # Feed-forward layers.
    out = self.ff_layer1(attn_out)
    out = self.ff_layer2(out)
    out = self.ff_dropout(out, training=training)
    out = self.ff_ln(out + attn_out)

    if self.flatten:
      out = tf.reshape(out, h_shape)

    input_batch['hidden_vecs'] = out
    input_batch[f'attn_scores/{self.name}'] = attn_score

    return input_batch


class SelfAttnModeTransformerLayer(tf.keras.layers.Layer):
  """Performs full self-attention across the future modes dimensions."""

  def __init__(self,
               num_heads=8,
               hidden_size=256,
               drop_prob=0.1,
               ln_eps=1e-6,
               ff_dim=128):
    super().__init__()
    self.hidden_size = hidden_size
    if hidden_size % num_heads != 0:
      raise ValueError(f'hidden_size ({hidden_size}) must be an integer '
                       f'times bigger than num_heads ({num_heads}).')
    self.attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        attention_axes=3)  # Attention over modes
    self.attn_dropout = tf.keras.layers.Dropout(drop_prob)
    self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f', output_shape=ff_dim, bias_axes='f', activation='relu')
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None)
    self.ff_dropout = tf.keras.layers.Dropout(drop_prob)
    self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

  def call(self, input_batch, training=None):
    input_batch = input_batch.copy()

    # [b, a, t, h] or [b, a, t, n, h]
    hidden_vecs = input_batch['hidden_vecs']

    attn_out, attn_score = self.attn_layer(
        query=hidden_vecs,
        key=hidden_vecs,
        value=hidden_vecs,
        attention_mask=None,
        return_attention_scores=True)
    out = self.attn_dropout(attn_out, training=training)
    attn_out = self.attn_ln(out + hidden_vecs)

    # Feed-forward layers.
    out = self.ff_layer1(attn_out)
    out = self.ff_layer2(out)
    out = self.ff_dropout(out, training=training)
    out = self.ff_ln(out + attn_out)

    input_batch['hidden_vecs'] = out
    input_batch[f'attn_scores/{self.name}'] = attn_score

    return input_batch


class SceneNonMultimodalCrossAttnTransformerLayer(tf.keras.layers.Layer):
  """Performs cross-attention between the occupancy grid and agents."""

  def __init__(self,
               num_heads=8,
               hidden_size=256,
               drop_prob=0.0,
               ln_eps=1e-6,
               ff_dim=128):
    super().__init__()
    self.hidden_size = hidden_size
    if hidden_size % num_heads != 0:
      raise ValueError(f'hidden_size ({hidden_size}) must be an integer times'
                       f' bigger than num_heads ({num_heads}).')
    self.attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        attention_axes=3)
    self.attn_dropout = tf.keras.layers.Dropout(drop_prob)
    self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f', output_shape=ff_dim, bias_axes='f', activation='relu')
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None)
    self.ff_dropout = tf.keras.layers.Dropout(drop_prob)
    self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

  def call(self, input_batch, training=None):
    input_batch = input_batch.copy()

    # [b, a, t, h]
    hidden_vecs = input_batch['hidden_vecs']

    # [b, N, H]
    scene_hidden_vec = input_batch['scene_hidden_vec']

    # [b, 1, 1, N, H]
    scene_hidden_vec = scene_hidden_vec[:, tf.newaxis, tf.newaxis]

    t = hidden_vecs.shape[2]
    a = hidden_vecs.shape[1]

    scene_hidden_vec = tf.tile(scene_hidden_vec, [1, a, t, 1, 1])

    # [b, a, t, 1, h]
    hidden_vecs_ext = hidden_vecs[..., tf.newaxis, :]

    # No need for mask since there is only 1 element for key/value.
    attn_out, attn_score = self.attn_layer(
        query=hidden_vecs_ext,
        key=scene_hidden_vec,
        value=scene_hidden_vec,
        attention_mask=None,
        return_attention_scores=True)
    out = self.attn_dropout(attn_out, training=training)[..., 0, :]
    attn_out = self.attn_ln(out + hidden_vecs)
    out = self.ff_layer1(attn_out)
    out = self.ff_layer2(out)
    out = self.ff_dropout(out, training=training)
    out = self.ff_ln(out + attn_out)

    input_batch['hidden_vecs'] = out

    input_batch[f'attn_scores/{self.name}'] = attn_score
    return input_batch


class SceneMultimodalCrossAttnTransformerLayer(tf.keras.layers.Layer):
  """Performs cross-attention between the occupancy grid and agents."""

  def __init__(self,
               num_heads=8,
               hidden_size=256,
               drop_prob=0.0,
               ln_eps=1e-6,
               ff_dim=128):
    super().__init__()
    self.hidden_size = hidden_size
    if hidden_size % num_heads != 0:
      raise ValueError(f'hidden_size ({hidden_size}) must be an integer times'
                       f' bigger than num_heads ({num_heads}).')
    self.attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        attention_axes=(4,))
    self.attn_dropout = tf.keras.layers.Dropout(drop_prob)
    self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f', output_shape=ff_dim, bias_axes='f', activation='relu')
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None)
    self.ff_dropout = tf.keras.layers.Dropout(drop_prob)
    self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

  def call(self, input_batch, training=None):
    input_batch = input_batch.copy()

    # [b, a, t, n, h]
    hidden_vecs = input_batch['hidden_vecs']
    # [b, N, H]
    scene_hidden_vec = input_batch['scene_hidden_vec']

    # [b, 1, 1, 1, N, H]
    scene_hidden_vec = scene_hidden_vec[:, tf.newaxis, tf.newaxis, tf.newaxis]

    t = hidden_vecs.shape[2]
    a = hidden_vecs.shape[1]
    n = hidden_vecs.shape[3]

    scene_hidden_vec = tf.tile(scene_hidden_vec, [1, a, t, n, 1, 1])

    # [b, a, t, n, 1, h]
    hidden_vecs_ext = hidden_vecs[..., tf.newaxis, :]

    # No need for mask since there is only 1 element for key/value.
    attn_out, attn_score = self.attn_layer(
        query=hidden_vecs_ext,
        key=scene_hidden_vec,
        value=scene_hidden_vec,
        attention_mask=None,
        return_attention_scores=True)
    out = self.attn_dropout(attn_out, training=training)[..., 0, :]
    attn_out = self.attn_ln(out + hidden_vecs)
    out = self.ff_layer1(attn_out)
    out = self.ff_layer2(out)
    out = self.ff_dropout(out, training=training)
    out = self.ff_ln(out + attn_out)

    input_batch['hidden_vecs'] = out
    input_batch[f'attn_scores/{self.name}'] = attn_score
    return input_batch
