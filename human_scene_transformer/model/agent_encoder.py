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

"""Contains agent encoder layers."""

from typing import Dict, Optional

from human_scene_transformer.model.agent_feature_encoder import AgentTemporalEncoder
from human_scene_transformer.model.model_params import ModelParams

import tensorflow as tf


class FeatureConcatAgentEncoderLayer(tf.keras.layers.Layer):
  """Independently encodes features and attends to them.

  Agent features are cross-attended with a learned query or hidden_vecs instead
  of MLP.
  """

  def __init__(self,
               params: ModelParams):
    super().__init__()

    # Cross Attention and learned query.
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=params.hidden_size,
        bias_axes='h',
        activation=None,
    )
    self.ff_dropout = tf.keras.layers.Dropout(params.drop_prob)

    self.agent_feature_embedding_layers = []
    # Position Feature
    self.agent_feature_embedding_layers.append(
        params.agents_feature_config[params.agents_position_key](
            params.agents_position_key, params.hidden_size - 8, params
        )
    )
    # Feature Embedding
    for key, layer in params.agents_feature_config.items():
      if key == params.agents_position_key:
        continue
      self.agent_feature_embedding_layers.append(
          layer(key, params.hidden_size - 8, params)
      )

    # Temporal Embedding
    self.agent_feature_embedding_layers.append(
        AgentTemporalEncoder(
            list(params.agents_feature_config.keys())[0],
            params.hidden_size - 8,
            params,
        )
    )

  def call(
      self, input_batch: Dict[str, tf.Tensor], training: Optional[bool] = None
  ):
    input_batch = input_batch.copy()

    layer_embeddings = []
    for layer in self.agent_feature_embedding_layers:
      layer_embedding, _ = layer(input_batch, training=training)
      layer_embedding = tf.reshape(
          layer_embedding,
          layer_embedding.shape[:-2]
          + [layer_embedding.shape[-2] * layer_embedding.shape[-1]],
      )
      layer_embeddings.append(layer_embedding)
    embedding = tf.concat(layer_embeddings, axis=-1)

    out = self.ff_layer2(embedding)

    input_batch['hidden_vecs'] = out
    input_batch['hidden_vecs_fe'] = out
    return input_batch


class FeatureAttnAgentEncoderLearnedLayer(tf.keras.layers.Layer):
  """Independently encodes features and attends to them.

  Agent features are cross-attended with a learned query or hidden_vecs instead
  of MLP.
  """

  def __init__(self,
               params: ModelParams):
    super(FeatureAttnAgentEncoderLearnedLayer, self).__init__()

    # Cross Attention and learned query.
    self.attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=params.num_heads,
        key_dim=params.hidden_size,  # "large" to prevent a bottleneck
        attention_axes=3)
    self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=params.ln_eps)
    self.ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f',
        output_shape=params.transformer_ff_dim,
        bias_axes='f',
        activation='relu')
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=params.hidden_size,
        bias_axes='h',
        activation=None)
    self.ff_dropout = tf.keras.layers.Dropout(params.drop_prob)
    self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=params.ln_eps)

    self.agent_feature_embedding_layers = []
    # Position Feature
    self.agent_feature_embedding_layers.append(
        params.agents_feature_config[params.agents_position_key](
            params.agents_position_key, params.hidden_size-8, params))
    # Feature Embedding
    for key, layer in params.agents_feature_config.items():
      if key == params.agents_position_key:
        continue
      self.agent_feature_embedding_layers.append(
          layer(key, params.hidden_size-8, params))

    # Temporal Embedding
    self.agent_feature_embedding_layers.append(
        AgentTemporalEncoder(list(params.agents_feature_config.keys())[0],
                             params.hidden_size-8, params))

    # [1, 1, 1, 1, h]
    self.learned_query_vec = tf.Variable(
        tf.random_uniform_initializer(
            minval=-1., maxval=1.)(shape=[1, 1, 1, 1, params.hidden_size]),
        trainable=True,
        dtype=tf.float32)

  def _build_learned_query(self, input_batch):
    """Converts self.learned_query_vec into a learned query vector."""
    # This weird thing is for exporting and loading keras model...
    b = tf.shape(input_batch['agents/position'])[0]
    num_agents = tf.shape(input_batch['agents/position'])[1]
    num_steps = tf.shape(input_batch['agents/position'])[2]

    # [b, num_agents, num_steps, 1, h]
    return tf.tile(self.learned_query_vec, [b, num_agents, num_steps, 1, 1])

  def call(self, input_batch: Dict[str, tf.Tensor],
           training: Optional[bool] = None):
    input_batch = input_batch.copy()

    layer_embeddings = []
    layer_masks = []
    for layer in self.agent_feature_embedding_layers:
      layer_embedding, layer_mask = layer(input_batch, training=training)
      layer_embeddings.append(layer_embedding)
      layer_masks.append(layer_mask)
    embedding = tf.concat(layer_embeddings, axis=3)

    b = tf.shape(embedding)[0]
    a = tf.shape(embedding)[1]
    t = tf.shape(embedding)[2]
    n = tf.shape(embedding)[3]

    # [1, 1, 1, N, 8]
    one_hot = tf.one_hot(tf.range(0, n), 8)[None, None, None]
    # [b, a, t, N, 8]
    one_hot_id = tf.tile(one_hot, (b, a, t, 1, 1))

    embedding = tf.concat([embedding, one_hot_id], axis=-1)

    attention_mask = tf.concat(layer_masks, axis=-1)

    # [b, a, t, num_heads, 1, num_features] <- broadcasted
    # Newaxis for num_heads, num_features
    attention_mask = attention_mask[..., tf.newaxis, tf.newaxis, :]

    learned_query = self._build_learned_query(input_batch)

    # Attention along axis 3
    attn_out, attn_score = self.attn_layer(
        query=learned_query,
        key=embedding,
        value=embedding,
        attention_mask=attention_mask,
        return_attention_scores=True)
    # [b, a, t, h]
    attn_out = attn_out[..., 0, :]
    out = self.ff_layer1(attn_out)
    out = self.ff_layer2(out)
    out = self.ff_dropout(out, training=training)
    out = self.ff_ln(out + attn_out)

    input_batch['hidden_vecs'] = out
    input_batch[f'attn_scores/{self.name}'] = attn_score
    return input_batch
