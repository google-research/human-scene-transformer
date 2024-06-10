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

"""Model parameter classes.

The module contains a dataclasses used to configure the model.
"""

import gin

from human_scene_transformer import is_hidden_generators
from human_scene_transformer.model import agent_feature_encoder
from human_scene_transformer.model import head
from human_scene_transformer.model import scene_encoder as _


@gin.configurable
class ModelParams(object):
  """This object configures the model."""

  def __init__(
      self,
      agents_feature_config=None,
      agents_position_key='agents/position',
      agents_orientation_key='agents/orientation',
      hidden_size=128,
      feature_embedding_size=128,
      transformer_ff_dim=64,
      ln_eps=1e-6,
      num_heads=4,
      num_conv_filters=(32, 32, 64, 64, 128),
      num_modes=4,
      attn_architecture=(
          'self-attention',
          'cross-attention',
          'multimodality_induction',
          'self-attention',
          'self-attention-mode',
          'self-attention',
          'self-attention-mode',
      ),
      mask_style=None,
      scene_encoder=None,
      prediction_head=head.PredictionHeadLayer,
      prediction_head_hidden_units=None,
      drop_prob=0.1,
      num_history_steps=12,
      num_steps=49,
      timestep=1 / 6,
      is_hidden_generator=is_hidden_generators.BPIsHiddenGenerator,
  ):
    """Initialize the TrainParam object.

    This is where gin injection occurs.

    Args:
      agents_feature_config: Dict mapping input features to encoders.
      agents_position_key: Name of the position key.
      agents_orientation_key: Name of the orientation key.
      hidden_size: The hidden vector size of the transformer.
      feature_embedding_size: The embedding vector size of agent features.
      transformer_ff_dim: The hidden layer size of transformer layers.
      ln_eps: Epsilon value for layer norm operations.
      num_heads: The number of heads in multi-headed attention.
      num_conv_filters: The number of filters in the occupancy grid convolution
        layers.
      num_modes: Number of predicted possible future outcomes. Each agent has
        exactly one predicted trajectory per possible outcome.
      attn_architecture: A tuple of strings describing the transformer
        architecture. Options are: 'self-attention', 'cross-attention' and
        'multimodality_induction'.
      mask_style: A string or None describing the mask in self-attention.
        'fully_padded' means timesteps of padded agents will not participate in
        the attention computation. 'any_padded' means any padded timesteps will
        not participate in the attention computation.
      scene_encoder: The scene encoder to use.
      prediction_head: Type of prediction head to use: 'full' with all outputs
        or '2d_pos' where only the 2D position is predicted.
      prediction_head_hidden_units: # A tuple describing the number of hidden
        units in the Prediction head. Set to None to disable the hidden layer.
      drop_prob: Dropout probability during training.
      num_history_steps: Number of history timesteps per data point.
      num_steps: Number of timesteps per data point.
      timestep: Delta time of a timestep in seconds.
      is_hidden_generator: The class to generating the is_hidden tensor. When
        called, the generator class returns a np array representing which
        agents and timesteps should be masked from the model.
    """
    if agents_feature_config is None:
      self.agents_feature_config = {
          'agents/position': agent_feature_encoder.AgentPositionEncoder,
          'agents/orientation': agent_feature_encoder.Agent2DOrientationEncoder,
          'agents/detection_score': agent_feature_encoder.AgentScalarEncoder,
          'agents/detection_stage': agent_feature_encoder.AgentOneHotEncoder
          }
    else:
      self.agents_feature_config = agents_feature_config
    self.agents_position_key = agents_position_key
    self.agents_orientation_key = agents_orientation_key
    self.hidden_size = hidden_size
    self.feature_embedding_size = feature_embedding_size
    self.transformer_ff_dim = transformer_ff_dim
    self.ln_eps = ln_eps
    self.num_heads = num_heads
    self.num_conv_filters = num_conv_filters
    self.num_modes = num_modes
    self.attn_architecture = attn_architecture
    self.mask_style = mask_style
    self.scene_encoder = scene_encoder
    self.prediction_head = prediction_head
    self.prediction_head_hidden_units = prediction_head_hidden_units
    self.drop_prob = drop_prob
    self.is_hidden_generator = is_hidden_generator

    self.num_history_steps = num_history_steps
    self.num_steps = num_steps
    self.timestep = timestep
