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

"""is_hidden generator classes for deciding HST operation mode.

Currently includes BP (predicting all agents), Conditional-BP (predicting
agents conditioned on the 0th agent, usually the ego-robot).
"""

import gin
import numpy as np


@gin.configurable
class IsHiddenGenerator(object):
  """Base class generator for the is_hidden tensor."""

  def __init__(self, num_steps, num_history_steps):
    self._num_steps = num_steps
    self._num_history_steps = num_history_steps

  def __call__(self, num_agents: int, train_progress: float = 0.0):
    """Returns the is_hidden tensor.

    Args:
      num_agents: Number of agents in the scene.
      train_progress: A float between 0 to 1 representing the overall progress
        of training. This float can be current_step / total_training_steps. This
        float can be used for training w/ an annealing schedule.
    """
    raise NotImplementedError('Calling the base class is prohibited.')


@gin.configurable
class BPIsHiddenGenerator(IsHiddenGenerator):
  """is_hidden generator for BP. All agents futures are hidden."""

  def __call__(self, num_agents: int, train_progress: float = 0.0):
    """Returns the is_hidden tensor for behavior prediction.

    Always returns 0 (not hidden) for history/current steps and 1 (hidden)
    for future steps.

    Args:
      num_agents: Number of agents in the scene.
      train_progress: A float between 0 to 1 representing the overall progress
        of training. This float can be current_step / total_training_steps. This
        float can be used for training w/ an annealing schedule.

    Returns:
      is_hidden: The is_hidden tensor for behavior prediction.
    """

    # [1, a, t, 1].
    is_hidden = np.ones([1, num_agents, self._num_steps, 1],
                        dtype=bool)
    is_hidden[:, :, :self._num_history_steps + 1, :] = False
    return is_hidden


@gin.configurable
class CBPIsHiddenGenerator(IsHiddenGenerator):
  """is_hidden generator for Conditional-BP."""

  def __call__(self, num_agents: int, train_progress: float = 0.0):
    """Returns the is_hidden tensor for conditional behavior prediction.

    Always returns 0 (not hidden) for history/current steps and 1 (hidden) for
    future steps except for the 0th agent, which is all 0. This is often used
    along with ModelParams.robot_as_0th_agent=True so that the model can predict
    how other agents move conditioned on a given robot trajectory.

    Args:
      num_agents: Number of agents in the scene.
      train_progress: A float between 0 to 1 representing the overall progress
        of training. This float can be current_step / total_training_steps. This
        float can be used for training w/ an annealing schedule.

    Returns:
      is_hidden: The is_hidden tensor for conditional behavior prediction.
    """

    # [1, a, t, 1].
    is_hidden = np.ones([1, num_agents, self._num_steps, 1],
                        dtype=bool)
    is_hidden[:, :, :self._num_history_steps + 1, :] = False
    is_hidden[:, 0, :, :] = False
    return is_hidden
