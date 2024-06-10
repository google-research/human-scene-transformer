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
class PedestriansDatasetParams(object):
  """This object describes parameters of the dataset.

  The default values represent the proxy dataset.
  """

  def __init__(
      self,
      path=None,
      dataset='eth',
      train_config='train',
      eval_config='val',
      num_history_steps=7,
      num_steps=20,
      num_agents=16,
      timestep=0.4,
  ):

    self.path = path
    self.dataset = dataset
    self.train_config = train_config
    self.eval_config = eval_config
    self.num_history_steps = num_history_steps
    self.num_steps = num_steps
    self.num_agents = num_agents
    self.timestep = timestep
