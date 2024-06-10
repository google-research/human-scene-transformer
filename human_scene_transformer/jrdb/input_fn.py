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
from typing import List

from human_scene_transformer.jrdb.dataset_params import JRDBDatasetParams

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry import transformation


def _map_to_expected_input_format(dp):
  """Maps the output of the JRDB Dataset pipeline to HST input format."""
  dp['agents/position'] = dp['agents/position'][..., :2]
  ears_vec = dp['agents/keypoints'][:, :, 8] - dp['agents/keypoints'][:, :, 7]
  dp['agents/gaze'] = _wrap_angle(
      tf.math.atan2(ears_vec[..., 1], ears_vec[..., 0]) + np.pi / 2
  )[..., tf.newaxis]

  # Reshape keypoints from [a, t, 33, 3] to [a, t, 33 * 3 = 99]
  shape = [tf.shape(dp['agents/keypoints'])[k] for k in range(4)]
  dp['agents/keypoints'] = tf.reshape(dp['agents/keypoints'],
                                      [shape[0], shape[1], shape[2] * shape[3]])

  return dp


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
      paddings, [[axis, 0], [axis, 1]], [pad_beginning, pad_end], name=None
  )
  padded_tensor = tf.pad(
      tensor, paddings=paddings, mode='CONSTANT', constant_values=value
  )
  return padded_tensor


def _get_random_window(
    feature_dict,
    idx_start,
    split_stop=1.0,
    window_length=35,
    target_num_agents=32,
    translate_to_robot_origin=True,
    max_pad_beginning=8,
    random_focus_agent=False,
    min_distance_to_robot=7.0,
):
  """Extracts a random window of fixed length from the scene.

  Args:
    feature_dict: Feature dictionary of full scene.
    idx_start: Index at which the window is started.
    split_stop: Split stop fraction.
    window_length: Length of extracted window.
    target_num_agents: Number of agents per data point. If there are not enough
      agents they will be padded. If there are too many agents the closest
      agents to the robot will be selected. If no robot information is available
      random agents will be picked.
    translate_to_robot_origin: If True the datapoint will be shifted such that
      the first robot position is at (0, 0).
    max_pad_beginning: Maximum number of padded positions at the beginning of a
      tensor. Typically num_history - 1 to have at least one historic value.
    random_focus_agent: Select a random agent and select the closest
      target_num_agents around the agent.
    min_distance_to_robot: Filter out agents with larger
        minimum distance to robot in selected window.

  Returns:
    Datapoint
  """

  features = feature_dict.keys()
  agents_position = feature_dict['agents/position']
  scene_length = agents_position.bounding_shape(axis=1)

  split_stop_index = tf.cast(
      tf.math.ceil(split_stop * tf.cast(scene_length, tf.float32)), tf.int64
  )

  num_pad_start = tf.random.uniform(
      (), 0, tf.cast(max_pad_beginning + 1, tf.int64), dtype=tf.int64
  )

  rand_idx_stop = tf.math.minimum(
      idx_start + window_length - num_pad_start, split_stop_index
  )

  num_pad = window_length - (rand_idx_stop - idx_start) - num_pad_start

  num_pad_end = num_pad

  idx_slice = slice(idx_start, rand_idx_stop)

  if translate_to_robot_origin:
    robot_origin = feature_dict['robot/position'][idx_start][tf.newaxis]

  output_dict = dict()

  # [all agents in scene]
  agents_present_in_window = (
      tf.reduce_sum(
          tf.reduce_sum(tf.abs(agents_position[:, idx_slice]), axis=-1), axis=1
      )
      > 0
  )

  distance_vec = (
      agents_position[:, idx_slice].to_tensor(np.inf)
      - feature_dict['robot/position'][tf.newaxis, idx_slice]
  )
  distance = tf.sqrt(tf.reduce_sum(tf.square(distance_vec[..., :3]), axis=-1))

  distance_mask = tf.reduce_any(distance < min_distance_to_robot, axis=-1)

  agents_present_in_window = tf.logical_and(
      agents_present_in_window, distance_mask
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

  if (target_num_agents is not None
      and not tf.reduce_any(agents_present_in_window)):
    selected_agents_position = (
        tf.ones([target_num_agents, window_length, 3]) * np.nan
    )
    selected_indices = tf.range(1, dtype=tf.int32)
  elif not tf.reduce_any(agents_present_in_window):
    selected_agents_position = (
        tf.ones([1, window_length, 3]) * np.nan
    )
    selected_indices = tf.range(1, dtype=tf.int32)
  else:
    # [a, t, 3]
    agents_position_present_full_tensor = (
        agents_position_present[:, idx_slice]
    ).to_tensor(np.nan)

    # Pad number of agents to target_num_agents
    if target_num_agents is not None and num_agents > target_num_agents:
      if 'robot/position' in features and not random_focus_agent:
        # Get closest agents to robot
        distance_vec = (
            agents_position_present_full_tensor
            - feature_dict['robot/position'][tf.newaxis, idx_slice]
        )
        distance = tf.linalg.norm(distance_vec, axis=-1)
        mean_distance = tf.experimental.numpy.nanmean(distance, axis=-1)
        sorted_distance_idx = tf.argsort(mean_distance)
        selected_indices = sorted_distance_idx[:target_num_agents]
        selected_agents_position = tf.gather(
            agents_position_present_full_tensor, selected_indices
        )
      elif random_focus_agent:
        focus_agent_idx = tf.random.uniform(
            (), maxval=num_agents, dtype=tf.int64
        )
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
        selected_indices = sorted_distance_idx[: 4 * target_num_agents]

        selected_indices = tf.random.shuffle(selected_indices)[
            :target_num_agents
        ]
        selected_agents_position = tf.gather(
            agents_position_present_full_tensor, selected_indices
        )

      else:
        sorted_indices = tf.range(num_agents, dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(sorted_indices)
        selected_indices = shuffled_indices[:target_num_agents]
        selected_agents_position = tf.gather(
            agents_position_present_full_tensor, selected_indices
        )
    elif target_num_agents is not None and  num_agents < target_num_agents:
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

    if translate_to_robot_origin:
      selected_agents_position = (
          selected_agents_position - robot_origin[tf.newaxis]
      )

    selected_agents_position = _pad(
        selected_agents_position, num_pad_start, num_pad_end_agents, axis=1
    )

  if target_num_agents is not None:
    output_dict['agents/position'] = tf.ensure_shape(
        selected_agents_position, [target_num_agents, window_length, 3]
    )
  else:
    output_dict['agents/position'] = selected_agents_position

  #######################
  # Agents Orientation #
  ######################
  if 'agents/orientation' in features and tf.reduce_any(
      agents_present_in_window
  ):
    agents_orientation = feature_dict['agents/orientation']
    agents_orientation_present = tf.gather(
        agents_orientation, tf.where(agents_present_in_window)[:, 0]
    )

    # [a, t, 1]
    agents_orientation_present_full_tensor = (
        agents_orientation_present[:, idx_slice]
    ).to_tensor(np.nan)

    # Pad number of agents to target_num_agents
    if target_num_agents is not None and num_agents > target_num_agents:
      selected_agents_orientation = tf.gather(
          agents_orientation_present_full_tensor, selected_indices
      )
    elif target_num_agents is not None and num_agents < target_num_agents:
      selected_agents_orientation = _pad(
          agents_orientation_present_full_tensor,
          0,
          target_num_agents - num_agents,
          axis=0,
          value=np.nan,
      )
    else:
      selected_agents_orientation = agents_orientation_present_full_tensor

    selected_agents_orientation = _pad(
        selected_agents_orientation, num_pad_start, num_pad_end_agents, axis=1
    )

    if target_num_agents is not None:
      output_dict['agents/orientation'] = tf.ensure_shape(
          selected_agents_orientation, [target_num_agents, window_length, 1]
      )
    else:
      output_dict['agents/orientation'] = selected_agents_orientation
  else:
    if target_num_agents is not None:
      output_dict['agents/orientation'] = (
          tf.ones([target_num_agents, window_length, 1]) * np.nan
      )
    else:
      output_dict['agents/orientation'] = (
          tf.ones([1, window_length, 1]) * np.nan
      )

  #####################
  # Agents Keypoints #
  ####################
  if 'agents/keypoints' in features and tf.reduce_any(agents_present_in_window):
    agents_keypoints = feature_dict['agents/keypoints']
    agents_keypoints_present = tf.gather(
        agents_keypoints, tf.where(agents_present_in_window)[:, 0]
    )

    # [a, t, 1]
    agents_keypoints_present_full_tensor = (
        agents_keypoints_present[:, idx_slice]
    ).to_tensor(np.nan)

    # Pad number of agents to target_num_agents
    if  target_num_agents is not None and num_agents > target_num_agents:
      selected_agents_keypoints = tf.gather(
          agents_keypoints_present_full_tensor, selected_indices
      )
    elif  target_num_agents is not None and num_agents < target_num_agents:
      selected_agents_keypoints = _pad(
          agents_keypoints_present_full_tensor,
          0,
          target_num_agents - num_agents,
          axis=0,
      )
    else:
      selected_agents_keypoints = agents_keypoints_present_full_tensor

    selected_agents_keypoints = _pad(
        selected_agents_keypoints, num_pad_start, num_pad_end_agents, axis=1
    )

    if  target_num_agents is not None:
      output_dict['agents/keypoints'] = tf.ensure_shape(
          selected_agents_keypoints, [target_num_agents, window_length, 33, 3]
      )
    else:
      output_dict['agents/keypoints'] = selected_agents_keypoints
  else:
    if  target_num_agents is not None:
      output_dict['agents/keypoints'] = (
          tf.ones([target_num_agents, window_length, 33, 3]) * np.nan
      )
    else:
      output_dict['agents/keypoints'] = (
          tf.ones([1, window_length, 33, 3]) * np.nan
      )

  ###################
  # Robot Position #
  ##################
  if 'robot/position' in features:
    robot_position = feature_dict['robot/position']

    # [t, 3]
    if translate_to_robot_origin:
      robot_position_window = robot_position[idx_slice] - robot_origin
    else:
      robot_position_window = robot_position[idx_slice]

    robot_position_window = _pad(
        robot_position_window, num_pad_start, num_pad_end, axis=0
    )

    output_dict['robot/position'] = tf.ensure_shape(
        robot_position_window, [window_length, 3]
    )
  ######################
  # Robot Orientation #
  #####################
  if 'robot/orientation' in features:
    robot_orientation = feature_dict['robot/orientation']

    robot_orientation_window = robot_orientation[idx_slice]

    robot_orientation_window = _pad(
        robot_orientation_window, num_pad_start, num_pad_end, axis=0
    )

    # [t, 1]
    output_dict['robot/orientation'] = tf.ensure_shape(
        robot_orientation_window, [window_length, 1]
    )

  ###############
  # Pointcloud #
  ##############
  if 'scene/pc' in features:
    num_p = 4096
    scene_pc = feature_dict['scene/pc']

    scene_pc_window = scene_pc[idx_slice]

    num_points_per_timestep = tf.shape(scene_pc_window)[1]

    if num_points_per_timestep > num_p:
      scelected_scene_pc_window = scene_pc_window[:, :num_p]
    elif num_points_per_timestep < num_p:
      scelected_scene_pc_window = _pad(
          scene_pc_window, 0, num_p - num_points_per_timestep, axis=1
      )
    else:
      scelected_scene_pc_window = scene_pc_window[:, :num_p]

    scelected_scene_pc_window = _pad(
        scelected_scene_pc_window, num_pad_start, num_pad_end, axis=0
    )

    if translate_to_robot_origin:
      scelected_scene_pc_window = scelected_scene_pc_window - robot_origin

    # [t, num_p, 3]
    output_dict['scene/pc'] = tf.ensure_shape(
        scelected_scene_pc_window, [window_length, num_p, 3]
    )

  output_dict['scene/id'] = feature_dict['scene/id']
  output_dict['scene/timestamp'] = idx_start

  return output_dict


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
      output_dict['agents/position'], rot_mat
  )

  if 'agents/orientation' in feature_dict.keys():
    output_dict['agents/orientation'] = _wrap_angle(
        output_dict['agents/orientation'] + yaw
    )

  if 'agents/keypoints' in feature_dict.keys():
    output_dict['agents/keypoints'] = transformation.rotation_matrix_3d.rotate(
        output_dict['agents/keypoints'], rot_mat
    )

  if 'robot/position' in feature_dict.keys():
    output_dict['robot/position'] = transformation.rotation_matrix_3d.rotate(
        output_dict['robot/position'], rot_mat
    )

  if 'robot/orientation' in feature_dict.keys():
    output_dict['robot/orientation'] = _wrap_angle(
        output_dict['robot/orientation'] + yaw
    )

  if 'scene/pc' in feature_dict.keys():
    output_dict['scene/pc'] = transformation.rotation_matrix_3d.rotate(
        output_dict['scene/pc'], rot_mat
    )

  return output_dict


def _random_translate_dp(feature_dict):
  """Translate a data point randomly."""
  output_dict = feature_dict.copy()

  translation = tf.random.uniform((2,), -10.0, 10.0)
  translation = tf.concat([translation, tf.convert_to_tensor([0.0])], 0)

  output_dict['agents/position'] = output_dict['agents/position'] + translation

  if 'robot/position' in feature_dict.keys():
    output_dict['robot/position'] = output_dict['robot/position'] + translation

  if 'scene/pc' in feature_dict.keys():
    output_dict['scene/pc'] = output_dict['scene/pc'] + translation

  return output_dict


def _sample_pointcloud(feature_dict, num_pointcloud_points):
  output_dict = feature_dict.copy()
  if 'scene/pc' in feature_dict:
    pc = output_dict['scene/pc']
    output_dict['scene/pc'] = tf.random.shuffle(pc)[:, :num_pointcloud_points]
  return output_dict


def _ragged_pointcloud_to_tensor(feature_dict):
  output_dict = feature_dict.copy()

  if 'scene/pc' in feature_dict:
    pc = output_dict['scene/pc'].to_tensor(np.nan)
    output_dict['scene/pc'] = pc

  return output_dict


def _subsample(feature_dict, start, step=1):
  """Subsamples scene."""
  output_dict = feature_dict.copy()

  output_dict['agents/position'] = output_dict['agents/position'][
      :, start::step
  ]

  if 'agents/orientation' in feature_dict.keys():
    output_dict['agents/orientation'] = output_dict['agents/orientation'][
        :, start::step
    ]

  if 'agents/action' in feature_dict.keys():
    output_dict['agents/action'] = output_dict['agents/action'][:, start::step]

  if 'agents/keypoints' in feature_dict.keys():
    output_dict['agents/keypoints'] = output_dict['agents/keypoints'][
        :, start::step
    ]

  if 'robot/position' in feature_dict.keys():
    output_dict['robot/position'] = output_dict['robot/position'][start::step]

  if 'robot/orientation' in feature_dict.keys():
    output_dict['robot/orientation'] = output_dict['robot/orientation'][
        start::step
    ]

  if 'scene/pc' in feature_dict:
    output_dict['scene/pc'] = output_dict['scene/pc'][start::step]

  return output_dict


def _load_scene_with_features(path, scene, features):
  """Loads a JRDB Scene with specified features."""
  datasets_list = []
  for feature in features:
    raw_dataset = tf.data.experimental.load(
        os.path.join(path, scene, *tuple(feature.split('/'))),
        compression='GZIP' if 'pc' in feature else None,
    )
    tagged_dataset = raw_dataset.map(lambda x: {feature: x})  # pylint: disable=cell-var-from-loop
    datasets_list.append(tagged_dataset)
  datasets_list.append(tf.data.Dataset.from_tensors({'scene/id': scene}))

  # Zip features to single scene dataset
  scene_dataset = tf.data.Dataset.zip(tuple(datasets_list))
  scene_dataset = scene_dataset.map(
      lambda *dicts: dict(collections.ChainMap(*dicts))
  )

  # Pre-compute dense tensor for pointcloud
  scene_dataset = scene_dataset.map(_ragged_pointcloud_to_tensor)
  return scene_dataset.cache()


def _scene_start_indices(
    feature_dict, split_start=0.0, split_stop=1.0, shuffle=False
):
  """Returns a tensor of all possible start indices for a scene."""
  agents_position = feature_dict['agents/position']
  scene_length = agents_position.bounding_shape(axis=1)

  split_start_index = tf.cast(
      tf.math.ceil(split_start * tf.cast(scene_length, tf.float32)), tf.int64
  )
  split_stop_index = tf.cast(
      tf.math.ceil(split_stop * tf.cast(scene_length, tf.float32)), tf.int64
  )
  start_idx_range = tf.range(
      split_start_index, split_stop_index, dtype=tf.int64
  )

  if shuffle:
    start_idx_range = tf.random.shuffle(start_idx_range)

  return start_idx_range


def load_dataset(
    dataset_params: JRDBDatasetParams,
    scenes: List[str],
    augment: bool,
    shuffle: bool,
    split_start: float = 0.0,
    split_stop: float = 1.0,
    allow_parallel: bool = True,
    evaluation: bool = False,
    repeat: bool = True,
    keep_subsamples: bool = True,
):
  """Loads JRDB Dataset with specified scenes."""
  subsample = dataset_params.subsample

  scene_datasets = [None] * len(scenes) * (subsample if keep_subsamples else 1)
  for j, scene in enumerate(scenes):
    scene_dataset = _load_scene_with_features(
        dataset_params.path, scene, dataset_params.features
    )

    for i in range(subsample if keep_subsamples else 1):
      scene_dataset_sub = scene_dataset.map(
          functools.partial(_subsample, start=i, step=subsample)
      ).cache()

      scene_start_idx_dataset = scene_dataset_sub.map(
          functools.partial(
              _scene_start_indices,
              split_start=split_start,
              split_stop=split_stop,
              shuffle=shuffle,
          )
      ).unbatch()

      scene_dataset_sub = tf.data.Dataset.zip(
          (scene_dataset_sub.repeat(), scene_start_idx_dataset)
      )

      scene_dataset_sub = scene_dataset_sub.prefetch(
          tf.data.experimental.AUTOTUNE
      )

      scene_datasets[j + len(scenes) * i] = scene_dataset_sub

  # Merge all scenes
  rand_wind_fun = functools.partial(
      _get_random_window,
      split_stop=split_stop,
      window_length=dataset_params.num_steps,
      target_num_agents=dataset_params.num_agents if not evaluation else 128,
      max_pad_beginning=dataset_params.num_history_steps if augment else 0,
      random_focus_agent=shuffle,
      min_distance_to_robot=dataset_params.min_distance_to_robot,
  )

  dataset = tf.data.Dataset.from_tensor_slices(scene_datasets).interleave(
      lambda x: x, cycle_length=len(scene_datasets)
  )

  dataset = dataset.map(
      rand_wind_fun,
      num_parallel_calls=(
          tf.data.experimental.AUTOTUNE if allow_parallel else None
      ),
      deterministic=True,
  )

  dataset = dataset.map(
      functools.partial(
          _sample_pointcloud,
          num_pointcloud_points=dataset_params.num_pointcloud_points),
      num_parallel_calls=(
          tf.data.experimental.AUTOTUNE if allow_parallel else None
      ),
      deterministic=True,
  )

  if augment:
    dataset = dataset.map(
        _random_rotate_dp,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True,
    )

    dataset = dataset.map(
        _random_translate_dp,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=True,
    )

  dataset = dataset.map(
      _map_to_expected_input_format,
      num_parallel_calls=(
          tf.data.experimental.AUTOTUNE if allow_parallel else None
      ),
      deterministic=True,
  )

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
      dataset_params,
      dataset_params.train_scenes,
      augment=True,
      shuffle=True,
      split_start=dataset_params.train_split[0],
      split_stop=dataset_params.train_split[1],
  )

  ds_train = ds_train.shuffle(100)

  return ds_train


def get_eval_dataset(dataset_params):
  """Returns a eval tf.dataset object."""
  ds_eval = load_dataset(
      dataset_params,
      dataset_params.eval_scenes,
      augment=False,
      shuffle=True,
      split_start=dataset_params.eval_split[0],
      split_stop=dataset_params.eval_split[1],
      evaluation=True
  )

  return ds_eval
