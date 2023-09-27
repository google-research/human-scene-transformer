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

"""Output distribution functions to further process the model's raw output."""

import tensorflow as tf
import tensorflow_probability as tfp


def force_positive(x, eps=1e-6):
  return tf.keras.activations.elu(x) + 1. + eps


@tf.function
def to_positive_definite_scale_tril(logit_sigma):
  tril = tfp.math.fill_triangular(logit_sigma)
  scale_tril = tf.linalg.set_diag(
      tril,
      force_positive(tf.linalg.diag_part(tril)))
  return scale_tril


@tf.function
def to_concentration(raw_concentration):
  return tf.math.reciprocal(force_positive(raw_concentration))


def get_position_distribution(model_output):
  """Multivariate Normal distribution over position."""
  p_pos = tfp.distributions.MultivariateNormalTriL(
      loc=model_output['agents/position'],
      scale_tril=to_positive_definite_scale_tril(
          model_output['agents/position/raw_scale_tril']))

  return p_pos


def get_multimodal_position_distribution(model_output):
  """Multivariate Normal Mixture distribution over position."""
  p_pos = get_position_distribution(model_output)

  p_pos_mm = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(
          logits=model_output['mixture_logits']),
      components_distribution=p_pos)

  return p_pos_mm


def get_orientation_distribution(model_output):
  """VonMises distribution over the yaw orientation.

  VonMises has a concentration input -> high values lead to unbounded log
  likelihoods. Thus we use the reciprocal of the elu to force vanishing
  gradients for high concentrations.
  Args:
    model_output: Raw model output dictionary.
      Required keys: agents/orientation, agents/orientation/raw_concentration
  Returns:
    VanMises distribution.
  """
  # [b, a, t, n, 1]
  p_orientation = tfp.distributions.VonMises(
      loc=model_output['agents/orientation'][..., 0],
      concentration=to_concentration(
          model_output['agents/orientation/raw_concentration'][..., 0]))

  return p_orientation


def get_multimodal_orientation_distribution(model_output):
  """VonMises distribution over position."""
  p_orientation = get_orientation_distribution(model_output)

  p_orientation_mm = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(
          logits=model_output['mixture_logits']),
      components_distribution=p_orientation)

  return p_orientation_mm
