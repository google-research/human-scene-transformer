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

"""JRDB dataset parameter class.

The module contains a dataclasses used to configure the dataset.
"""

import gin


@gin.configurable
class JRDBDatasetParams(object):
  """This object describes parameters of the dataset.

  The default values represent the proxy dataset.
  """

  def __init__(
      self,
      path=None,
      train_scenes=None,
      eval_scenes=None,
      features=None,
      train_split=(0., 1.),
      eval_split=(0., 1.),
      num_history_steps=6,
      num_steps=19,
      num_agents=8,
      timestep=0.33333333333334,
      subsample=5,
      min_distance_to_robot=7.0,
      num_pointcloud_points=512,
  ):
    """Initialize the DatasetParam object.

    Args:
      path: Location of dataset(s).
      train_scenes: Scenes to use in the train dataset.
      eval_scenes: Scenes to use in the eval dataset.
      features: The features to load from dataset file.
      train_split: Split to use from the train scenes in the train dataset.
      eval_split: Split to use from the eval scenes in the eval dataset.
      num_history_steps: The number of history timesteps.
      num_steps: The total number of timesteps.
      num_agents: The max number of agents in the dataset.
      timestep: The interval between timesteps.
      subsample: Subsample JRDB dataset by this frequency.
      min_distance_to_robot: Filter out agents with larger
        minimum distance to robot in selected window.
      num_pointcloud_points: Number of sampled pointcloud points.
    """

    self.path = path
    self.train_scenes = train_scenes
    self.eval_scenes = eval_scenes
    self.features = features
    self.train_split = train_split
    self.eval_split = eval_split
    self.num_history_steps = num_history_steps
    self.num_steps = num_steps
    self.num_agents = num_agents
    self.timestep = timestep
    self.subsample = subsample
    self.min_distance_to_robot = min_distance_to_robot
    self.num_pointcloud_points = num_pointcloud_points
