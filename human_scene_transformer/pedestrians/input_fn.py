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

"""Utility functions for generating input from preprocessed JRDB data."""

import collections
import functools
import os

from human_scene_transformer.pedestrians.dataset_params import PedestriansDatasetParams

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_graphics.geometry import transformation


def _agents_pos_to_ragged_tensor(agents_df):
  tensor_list = []
  for _, df in agents_df.groupby('id'):
    dropped_df = df.droplevel(1, axis=0)
    r_tensor = tf.RaggedTensor.from_value_rowids(
        values=np.vstack(df['p'].values).flatten().astype(np.float32),
        value_rowids=np.tile(np.array(dropped_df.index),
                             (2, 1)).transpose().flatten())
    tensor_list.append(r_tensor)
  return tf.stack(tensor_list)


def _get_file_as_pos_ragged(f):
  pos_df = pd.read_csv(
      f, header=None, delimiter='\t', names=['timestep', 'id', 'x', 'y'])
  pos_df.timestep = pos_df.timestep.astype(int) // 10
  pos_df.timestep = pos_df.timestep - pos_df.timestep.min()
  pos_df.id = pos_df.id.astype(int)
  pos_df = pos_df.set_index(['timestep', 'id']).apply(
      lambda x: pd.Series({'p': np.array(x)}), axis=1)
  pos_ragged = _agents_pos_to_ragged_tensor(pos_df)
  return pos_ragged


def _load_scene(f):
  """Loads a ETH/UCY Scene."""
  pos_ragged = _get_file_as_pos_ragged(f)

  pos_dataset = tf.data.Dataset.from_tensors({'agents/position': pos_ragged})
  scene_id_dataset = tf.data.Dataset.from_tensors(
      {'scene/id': os.path.basename(f).split('.')[0]})

  datasets_list = [pos_dataset, scene_id_dataset]

  # Zip features to single scene dataset
  scene_dataset = tf.data.Dataset.zip(tuple(datasets_list))
  scene_dataset = scene_dataset.map(
      lambda *dicts: dict(collections.ChainMap(*dicts)))

  return scene_dataset.cache()


def _scene_start_indices(feature_dict, shuffle=False):
  """Returns a tensor of all possible start indices for a scene."""
  agents_position = feature_dict['agents/position']
  scene_length = agents_position.bounding_shape(axis=1)

  split_start_index = 0
  split_stop_index = tf.cast(scene_length, tf.int64)
  start_idx_range = tf.range(
      split_start_index, split_stop_index, dtype=tf.int64)

  if shuffle:
    start_idx_range = tf.random.shuffle(start_idx_range)

  return start_idx_range


def _get_random_window(feature_dict,
                       idx_start,
                       window_length=20,
                       target_num_agents=None,
                       max_pad_beginning=7,
                       random_focus_agent=False):
  """Extracts a random window of fixed length from the scene.

  Args:
    feature_dict: Feature dictionary of full scene.
    idx_start: Index at which the window is started.
    window_length: Length of extracted window.
    target_num_agents: Number of agents per data point. If there are not enough
      agents they will be padded. If there are too many agents the closest
      agents to the robot will be selected. If no robot information is available
      random agents will be picked.
    max_pad_beginning: Maximum number of padded positions at the beginning of a
      tensor. Typically num_history - 1 to have at least one historic value.
    random_focus_agent: Select a random agent and select the closest
      target_num_agents around the agent.

  Returns:
    Datapoint
  """
  agents_position = feature_dict['agents/position']
  scene_length = agents_position.bounding_shape(axis=1)

  split_stop_index = tf.cast(scene_length, tf.int64)

  num_pad_start = tf.random.uniform((),
                                    0,
                                    tf.cast(max_pad_beginning + 1, tf.int64),
                                    dtype=tf.int64)

  rand_idx_stop = tf.math.minimum(
      idx_start + window_length - num_pad_start, split_stop_index
  )

  num_pad = window_length - (rand_idx_stop - idx_start) - num_pad_start

  num_pad_end = num_pad

  idx_slice = slice(idx_start, rand_idx_stop)

  output_dict = dict()

  # [all agents in scene]
  agents_present_in_window = (
      tf.reduce_sum(
          tf.reduce_sum(tf.abs(agents_position[:, idx_slice]), axis=-1), axis=1
      )
      > 0
  )

  ####################
  # Agents Position #
  ###################
  # Ragged [agents in window, scene_length, 2]
  agents_position_present = tf.gather(
      agents_position, tf.where(agents_present_in_window)[:, 0]
  )

  idx_len = agents_position_present[:, idx_slice].bounding_shape()[1]

  if idx_len < (rand_idx_stop - idx_start):
    num_pad_end_agents = num_pad_end + (rand_idx_stop - idx_start - idx_len)
  else:
    num_pad_end_agents = num_pad_end

  num_agents = tf.reduce_sum(tf.cast(agents_present_in_window, tf.int64))

  if target_num_agents is not None and not tf.reduce_any(
      agents_present_in_window):
    output_dict['agents/position'] = tf.ones(
        [target_num_agents, window_length, 2]) * np.nan
    selected_indices = tf.range(1, dtype=tf.int32)

  else:
    # [a, t, 3]
    agents_position_present_full_tensor = (
        agents_position_present[:, idx_slice]
    ).to_tensor(np.nan)

    # Pad number of agents to target_num_agents
    if target_num_agents is not None and num_agents > target_num_agents:
      if random_focus_agent:
        focus_agent_idx = tf.random.uniform((),
                                            maxval=num_agents,
                                            dtype=tf.int64)
        focus_agent_position = agents_position_present_full_tensor[
            focus_agent_idx
        ]

        distance_vec = (
            agents_position_present_full_tensor
            - focus_agent_position[tf.newaxis]
        )
        distance = tf.linalg.norm(distance_vec, axis=-1)
        distance = tf.where(tf.math.is_nan(distance), tf.float32.max, distance)
        min_distance = tf.reduce_min(distance, axis=-1)
        sorted_distance_idx = tf.argsort(min_distance)
        selected_indices = sorted_distance_idx[:target_num_agents]
        selected_agents_position = tf.gather(
            agents_position_present_full_tensor, selected_indices)
      else:
        sorted_indices = tf.range(num_agents, dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(sorted_indices)
        selected_indices = shuffled_indices[:target_num_agents]
        selected_agents_position = tf.gather(
            agents_position_present_full_tensor, selected_indices)
    elif target_num_agents is not None and num_agents < target_num_agents:
      selected_indices = tf.convert_to_tensor([0], dtype=tf.int32)
      selected_agents_position = _pad(
          agents_position_present_full_tensor,
          0,
          target_num_agents - num_agents,
          axis=0,
          value=np.nan,
      )
    else:
      selected_indices = tf.convert_to_tensor([0])
      selected_agents_position = agents_position_present_full_tensor

    selected_agents_position = _pad(
        selected_agents_position, num_pad_start, num_pad_end_agents, axis=1
    )

    if target_num_agents is not None:
      output_dict['agents/position'] = tf.ensure_shape(
          selected_agents_position, [target_num_agents, window_length, 2])
    else:
      output_dict['agents/position'] = selected_agents_position

  output_dict['scene/id'] = feature_dict['scene/id']
  output_dict['scene/timestamp'] = idx_start

  return output_dict


def _pad(tensor, pad_beginning, pad_end, axis, value=np.nan):
  """Pads a tensor along given dimension.

  Args:
    tensor: Input tensor to be padded.
    pad_beginning: Number of padded elements at the beginning.
    pad_end: Number of padded elements at the end.
    axis: Axis along the tensor is padded.
    value: Padding value.

  Returns:
    Padded tensor.
  """
  paddings = tf.zeros((tf.rank(tensor), 2), dtype=tf.int32)
  paddings = tf.tensor_scatter_nd_update(
      paddings, [[axis, 0], [axis, 1]], [pad_beginning, pad_end], name=None)
  padded_tensor = tf.pad(
      tensor, paddings=paddings, mode='CONSTANT', constant_values=value)
  return padded_tensor


def _wrap_angle(angle):
  return (angle + np.pi) % (2 * np.pi) - np.pi


def _random_rotate_dp(feature_dict):
  """Rotates a data point randomly around the origin."""
  output_dict = feature_dict.copy()

  yaw = tf.random.uniform((), -np.pi, np.pi)
  # [3, 3]
  rot_mat = transformation.rotation_matrix_3d.from_euler(
      tf.stack([0.0, 0.0, yaw]))

  output_dict['agents/position'] = transformation.rotation_matrix_3d.rotate(
      tf.concat(
          [
              output_dict['agents/position'],
              tf.zeros(output_dict['agents/position'][..., 0:1].shape),
          ],
          axis=-1,
      ),
      rot_mat,
  )[..., :2]

  return output_dict


def _random_translate_dp(feature_dict):
  """Translate a data point randomly."""
  output_dict = feature_dict.copy()

  translation = tf.random.uniform((2,), -10., 10.)

  output_dict['agents/position'] = output_dict['agents/position'] + translation

  return output_dict


def _list_dir(path):
  files = os.listdir(path)
  return files


def load_dataset(dataset_params: PedestriansDatasetParams, split, augment,
                 shuffle, repeat=True, deterministic=False):
  """Loads Pedestrians Dataset with specified scenes."""
  scene_datasets = []
  if split == 'train':
    files = _list_dir(
        os.path.join(dataset_params.path, dataset_params.dataset, 'train'))
    files = [
        os.path.join(dataset_params.path, dataset_params.dataset, 'train', f)
        for f in files
    ]
    for f in files:
      scene_dataset = _load_scene(f)
      scene_start_idx_dataset = scene_dataset.map(
          functools.partial(_scene_start_indices, shuffle=shuffle)).unbatch()

      scene_dataset = tf.data.Dataset.zip(
          (scene_dataset.repeat(), scene_start_idx_dataset))

      scene_dataset = scene_dataset.prefetch(tf.data.experimental.AUTOTUNE)
      scene_datasets.append(scene_dataset)
    if dataset_params.train_config == 'trainval':
      files = _list_dir(
          os.path.join(dataset_params.path, dataset_params.dataset, 'val'))
      files = [
          os.path.join(dataset_params.path, dataset_params.dataset, 'val', f)
          for f in files
      ]
      for f in files:
        scene_dataset = _load_scene(f)
        scene_start_idx_dataset = scene_dataset.map(
            functools.partial(_scene_start_indices, shuffle=shuffle)).unbatch()
        scene_dataset = tf.data.Dataset.zip(
            (scene_dataset.repeat(), scene_start_idx_dataset))

        scene_dataset = scene_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        scene_datasets.append(scene_dataset)
  elif split == 'val':
    files = _list_dir(
        os.path.join(dataset_params.path, dataset_params.dataset, 'val'))
    files = [
        os.path.join(dataset_params.path, dataset_params.dataset, 'val', f)
        for f in files
    ]
    for f in files:
      scene_dataset = _load_scene(f)
      scene_start_idx_dataset = scene_dataset.map(
          functools.partial(_scene_start_indices, shuffle=shuffle)).unbatch()
      scene_dataset = tf.data.Dataset.zip(
          (scene_dataset.repeat(), scene_start_idx_dataset))

      scene_dataset = scene_dataset.prefetch(tf.data.experimental.AUTOTUNE)
      scene_datasets.append(scene_dataset)
  elif split == 'test':
    if dataset_params.eval_config == 'test':
      files = _list_dir(
          os.path.join(dataset_params.path, dataset_params.dataset, 'test'))
      files = [
          os.path.join(dataset_params.path, dataset_params.dataset, 'test', f)
          for f in files
      ]
    elif dataset_params.eval_config == 'val':
      files = _list_dir(
          os.path.join(dataset_params.path, dataset_params.dataset, 'val'))
      files = [
          os.path.join(dataset_params.path, dataset_params.dataset, 'val', f)
          for f in files
      ]
    for f in files:
      scene_dataset = _load_scene(f)
      scene_start_idx_dataset = scene_dataset.map(
          functools.partial(_scene_start_indices, shuffle=shuffle)).unbatch()
      scene_dataset = tf.data.Dataset.zip(
          (scene_dataset.repeat(), scene_start_idx_dataset))

      scene_dataset = scene_dataset.prefetch(tf.data.experimental.AUTOTUNE)
      scene_datasets.append(scene_dataset)

  rand_wind_fun = functools.partial(
      _get_random_window,
      window_length=dataset_params.num_steps,
      target_num_agents=dataset_params.num_agents,
      max_pad_beginning=max(
          (dataset_params.num_history_steps - 1), 0) if shuffle else 0,
      random_focus_agent=shuffle,
  )

  dataset = tf.data.Dataset.from_tensor_slices(scene_datasets).interleave(
      lambda x: x, cycle_length=len(scene_datasets))

  dataset = dataset.map(
      rand_wind_fun,
      num_parallel_calls=(
          tf.data.experimental.AUTOTUNE if not deterministic else None),
      deterministic=True)

  if augment:
    dataset = dataset.map(
        _random_rotate_dp,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True)

    dataset = dataset.map(
        _random_translate_dp,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True)

  if repeat:
    dataset = dataset.repeat()

  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = (
      tf.data.experimental.AutoShardPolicy.OFF
  )
  dataset = dataset.with_options(options)

  return dataset


def get_train_dataset(dataset_params):
  """Returns a training tf.dataset object."""
  ds_train = load_dataset(
      dataset_params, split='train', augment=True, shuffle=True)

  ds_train = ds_train.shuffle(1000)

  return ds_train


def get_eval_dataset(dataset_params):
  """Returns a eval tf.dataset object."""
  ds_eval = load_dataset(
      dataset_params, split='test', augment=False, shuffle=True)

  return ds_eval
