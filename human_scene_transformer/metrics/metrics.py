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

"""Entry point file for metrics."""

import gin

from human_scene_transformer.metrics import ade  # pylint: disable=unused-import
from human_scene_transformer.metrics import mean_angle_error  # pylint: disable=unused-import
from human_scene_transformer.metrics import pos_nll  # pylint: disable=unused-import

import tensorflow as tf


# Create a "stock" subclass from Keras' Mean for gin.
# Somehow, calling gin.external_configurable() did not work.
@gin.configurable
class Mean(tf.keras.metrics.Mean):

  def __init__(self, name="mean", dtype=None):
    super().__init__(name=name, dtype=dtype)
