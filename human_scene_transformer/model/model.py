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

"""Contains Human Scene Transformer keras model."""

from typing import Any, Dict, Optional, Tuple

from human_scene_transformer.model import agent_encoder
from human_scene_transformer.model import agent_self_alignment
from human_scene_transformer.model import attention
from human_scene_transformer.model import multimodality_induction
from human_scene_transformer.model import preprocess
from human_scene_transformer.model.model_params import ModelParams

import tensorflow as tf


class HumanTrajectorySceneTransformer(tf.keras.Model):
  """A variant of Scene Transformer adapted for human trajectory prediction.

  This class is a Keras model that is designed to predict the trajectory of
  bounding boxes of nearby humans (pedestrians). It is designed to be used on
  the proxy robots.
  The input of the model is an input_batch dict that maps input features to
  tensors. The exact features depends on the dataset (currently the EDR and
  Jackrobot dataset), but the input and output layers of the model will convert
  them to a dict with common feature names before passing to the transformer
  network.

  The model is a transformer that encodes each agent-timestep into a fixed size
  vector before doing self-attention and cross-attention w/ scene features such
  as the occupancy grid.
  The model outputs each agents future trajectories in the form of Gaussian
  distributions over the agents positions and von-Mises distributions over their
  corresponding orientations at each timestep. Furthermore, these distributions
  are multimodal similar to a Mixture Distributions but are shared for each
  agent with scene level mixture probabilities.

  """

  def __init__(self,
               params: ModelParams,
               input_layer: Optional[tf.keras.layers.Layer] = None,
               output_layer: Optional[tf.keras.layers.Layer] = None):
    """Constructs the model.

    Args:
      params: A model parameters object containing the model configuration.
      input_layer: A Keras Layer object that maps various features names in the
        dataset to a set of common names. E.g., tracks_tensors/xyz ->
        agent/positions
      output_layer: A Keras Layer object that maps various features predicted by
        the model to a set of names used in the dataset E.g., xyz ->
        agent/positions
    """

    super().__init__()
    self.input_layer = input_layer
    self.output_layer = output_layer

    # Preprocess layer.
    self.preprocess_layer = preprocess.PreprocessLayer(params)

    self.agent_encoding_layer = (
        agent_encoder.FeatureAttnAgentEncoderLearnedLayer(params)
    )

    if params.scene_encoder is not None:
      self.scene_encoder = params.scene_encoder(params)
    else:
      self.scene_encoder = None

    # Transformer layers.
    self.transformer_layers = []
    # Agent self alignment layer
    layer = agent_self_alignment.AgentSelfAlignmentLayer(
        hidden_size=params.hidden_size,
        ff_dim=params.transformer_ff_dim,
        num_heads=params.num_heads,
        mask_style='has_data')
    self.transformer_layers.append(layer)

    multimodality_induced = False
    for arch in params.attn_architecture:
      if arch == 'self-attention':
        layer = attention.SelfAttnTransformerLayer(
            hidden_size=params.hidden_size,
            ff_dim=params.transformer_ff_dim,
            num_heads=params.num_heads,
            drop_prob=params.drop_prob,
            mask_style=params.mask_style,
            flatten=True,
            multimodality_induced=multimodality_induced)
      elif arch == 'self-attention-mode':
        layer = attention.SelfAttnModeTransformerLayer(
            hidden_size=params.hidden_size,
            ff_dim=params.transformer_ff_dim,
            num_heads=params.num_heads,
            drop_prob=params.drop_prob)
      elif arch == 'cross-attention':
        if not multimodality_induced:
          layer = attention.SceneNonMultimodalCrossAttnTransformerLayer(
              hidden_size=params.hidden_size,
              ff_dim=params.transformer_ff_dim,
              num_heads=params.num_heads,
              drop_prob=params.drop_prob)
        else:
          layer = attention.SceneMultimodalCrossAttnTransformerLayer(
              hidden_size=params.hidden_size,
              ff_dim=params.transformer_ff_dim,
              num_heads=params.num_heads,
              drop_prob=params.drop_prob)
      elif arch == 'multimodality_induction':
        multimodality_induced = True
        layer = multimodality_induction.MultimodalityInduction(
            num_modes=params.num_modes,
            hidden_size=params.hidden_size,
            ff_dim=params.transformer_ff_dim,
            num_heads=params.num_heads,
            drop_prob=params.drop_prob)
      else:
        raise ValueError(f'Unknown attn architecture: {arch}. ' +
                         'Must be either self-attention or cross-attention.')
      self.transformer_layers.append(layer)
    # Prediction head.
    self.prediction_layer = params.prediction_head(
        hidden_units=params.prediction_head_hidden_units)

  def call(
      self,
      input_batch: Dict[str, tf.Tensor],
      is_hidden: Optional[Any] = None,
      training: bool = False) -> Tuple[Dict[str, tf.Tensor], Dict[str, Any]]:
    """Override the standard call() function to provide more flexibility.

    Args:
      input_batch: A dictionary that maps a str to a tensor. The tensor's first
        dimensions is always the batch dimension. These tensors include all
        timesteps (history, current and future) and all agents (observed and
        padded).
      is_hidden: An optional bool np array or tf.tensor of the shape: [batch,
        max_num_agents, num_time_steps, 1]. This tensor instructs the model on
        which agent-timestep needs to be predicted (if set to True). Note that
        you do not have to worry about padded agent or timesteps being
        predicted. The preprocessing layer will handle that for you. If you do
        not put in anything, a default behavior predction tensor will be used,
        which looks like: False for all agents in the history or current
        timesteps and True for all future timesteps.
      training: An optional bool that instructs the model if the call is used
        during training.

    Returns:
      output: A dict containing the model prediction. Note that the predicted
        tensors has the same shape as the input_batch so the history and
        current steps are included.
      input_batch: The input batch modified to include features generated by
        the preprocess layer, e.g., has_data.
    """

    if self.input_layer is not None:
      input_batch = self.input_layer(input_batch, training=training)

    # Preprocess input_batch.
    input_batch = self.preprocess_layer(input_batch, is_hidden=is_hidden)

    # Feed the input_batch through the network.
    input_batch = self.agent_encoding_layer(input_batch, training=training)
    if self.scene_encoder is not None:
      input_batch = self.scene_encoder(input_batch, training=training)
    for layer in self.transformer_layers:
      input_batch = layer(input_batch, training=training)
    predictions = self.prediction_layer(input_batch, training=training)
    if self.output_layer is not None:
      output = self.output_layer(predictions, training=training)
    else:
      output = predictions
    return output, input_batch
