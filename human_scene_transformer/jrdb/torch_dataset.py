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

"""PyTorch Dataset Wrapper for JRDB Dataset."""

from human_scene_transformer.jrdb import dataset_params as jrdb_dataset_params
from human_scene_transformer.jrdb import input_fn

import tensorflow_datasets as tfds

import torch


class JRDBPredictionDataset(torch.utils.data.IterableDataset):
  """PyTorch Dataset Wrapper for JRDB Dataset."""

  def __init__(self,
               dataset_params: jrdb_dataset_params.JRDBDatasetParams,
               train: bool = True):
    self.dataset_params = dataset_params
    self.train = train

  def __iter__(self):
    if self.train:
      tfds_iter = iter(
          tfds.as_numpy(input_fn.get_train_dataset(self.dataset_params)))
    else:
      tfds_iter = iter(
          tfds.as_numpy(input_fn.get_eval_dataset(self.dataset_params)))

    return tfds_iter
