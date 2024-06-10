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

"""Preprocesses the raw train split of JRDB.
"""

import collections
import json
import os

from absl import app
from absl import flags

from human_scene_transformer.data import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

_INPUT_PATH = flags.DEFINE_string(
    'input_path',
    default=None,
    help='Path to jrdb2022 dataset.'
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    default=None,
    help='Path to output folder.'
)

_PROCESS_POINTCLOUDS = flags.DEFINE_bool(
    'process_pointclouds',
    default=True,
    help='Whether to process pointclouds.'
)

_MAX_DISTANCE_TO_ROBOT = flags.DEFINE_float(
    'max_distance_to_robot',
    default=15.,
    help=('Maximum distance of agent to the robot to be included'
          ' in the processed dataset.')
)

_MAX_PC_DISTANCE_TO_ROBOT = flags.DEFINE_float(
    'max_pc_distance_to_robot',
    default=10.,
    help=('Maximum distance of pointcloud point to the robot to be included'
          ' in the processed dataset.')
)


AGENT_KEYPOINTS = True
FROM_DETECTIONS = True


def get_agents_dict(input_path, scene):
  """Returns agents GT data from raw data."""
  scene_data_file = utils.get_file_handle(
      os.path.join(input_path, 'labels', 'labels_3d', scene + '.json')
  )
  scene_data = json.load(scene_data_file)

  agents = collections.defaultdict(list)

  for frame in scene_data['labels']:
    ts = int(frame.split('.')[0])
    for det in scene_data['labels'][frame]:
      agents[det['label_id']].append((ts, det))
  return agents


def get_agents_dict_from_detections(input_path, scene):
  """Returns agents data from fused detections raw data."""
  scene_data_file = utils.get_file_handle(
      os.path.join(
          input_path, 'labels', 'labels_detections_3d', scene + '.json'
      )
  )
  scene_data = json.load(scene_data_file)

  agents = collections.defaultdict(list)

  for frame in scene_data['labels']:
    ts = int(frame.split('.')[0])
    for det in scene_data['labels'][frame]:
      agents[det['label_id']].append((ts, det))
  return agents


def get_agents_features(agents_dict, max_distance_to_robot=10):
  """Returns agents features from raw data dict."""
  agents_pos_dict = collections.defaultdict(dict)
  for agent_id, agent_data in agents_dict.items():
    for ts, agent_instance in agent_data:
      if agent_instance['attributes']['distance'] <= max_distance_to_robot:
        agents_pos_dict[(ts, agent_id)] = {
            'p': np.array([
                agent_instance['box']['cx'],
                agent_instance['box']['cy'],
                agent_instance['box']['cz'],
            ]),
            # rotation angle is relative to negative x axis of robot
            'yaw': np.pi - agent_instance['box']['rot_z'],
        }
  return agents_pos_dict


def jrdb_preprocess_train(input_path, output_path):
  """Preprocesses the raw train split of JRDB."""

  tf.keras.utils.set_random_seed(123)

  subsample = 1

  scenes = utils.list_scenes(
      os.path.join(input_path, 'train_dataset')
  )
  for scene in tqdm.tqdm(scenes):
    if not FROM_DETECTIONS:
      agents_dict = get_agents_dict(
          os.path.join(input_path, 'train_dataset'), scene
      )
    else:
      agents_dict = get_agents_dict_from_detections(
          os.path.join(input_path, 'processed'), scene
      )

    agents_features = utils.get_agents_features_with_box(
        agents_dict, max_distance_to_robot=_MAX_DISTANCE_TO_ROBOT.value
    )

    robot_odom = utils.get_robot(
        os.path.join(input_path, 'processed', 'odometry', 'train'), scene
    )

    agents_df = pd.DataFrame.from_dict(
        agents_features, orient='index'
    ).rename_axis(['timestep', 'id'])  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads

    if AGENT_KEYPOINTS:
      keypoints = utils.get_agents_keypoints(
          os.path.join(
              input_path, 'processed', 'labels',
              'labels_3d_keypoints', 'train'),
          scene,
      )
      keypoints_df = pd.DataFrame.from_dict(
          keypoints, orient='index'
      ).rename_axis(['timestep', 'id'])  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads

      agents_df = agents_df.join(keypoints_df)
      agents_df.keypoints.fillna(
          dict(
              zip(
                  agents_df.index[agents_df['keypoints'].isnull()],
                  [np.ones((33, 3)) * np.nan]
                  * len(
                      agents_df.loc[
                          agents_df['keypoints'].isnull(), 'keypoints'
                      ]
                  ),
              )
          ),
          inplace=True,
      )

    robot_df = pd.DataFrame.from_dict(robot_odom, orient='index').rename_axis(  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads
        ['timestep']
    )
    # Remove extra data odometry datapoints
    robot_df = robot_df.iloc[agents_df.index.levels[0]]

    assert (agents_df.index.levels[0] == robot_df.index).all()

    # Subsample
    assert len(agents_df.index.levels[0]) == agents_df.index.levels[0].max() + 1
    agents_df_subsampled_index = agents_df.unstack('id').iloc[::subsample].index
    agents_df = (
        agents_df.unstack('id')
        .iloc[::subsample]
        .reset_index(drop=True)
        .stack('id', dropna=True)
    )

    agents_in_odometry_df = utils.agents_to_odometry_frame(
        agents_df, robot_df.iloc[::subsample].reset_index(drop=True)
    )

    agents_pos_ragged_tensor = utils.agents_pos_to_ragged_tensor(
        agents_in_odometry_df
    )
    agents_yaw_ragged_tensor = utils.agents_yaw_to_ragged_tensor(
        agents_in_odometry_df
    )
    assert (
        agents_pos_ragged_tensor.shape[0] == agents_yaw_ragged_tensor.shape[0]
    )

    tf.data.Dataset.from_tensors(agents_pos_ragged_tensor).save(
        os.path.join(output_path, scene, 'agents', 'position')
    )
    tf.data.Dataset.from_tensors(agents_yaw_ragged_tensor).save(
        os.path.join(output_path, scene, 'agents', 'orientation')
    )

    if AGENT_KEYPOINTS:
      agents_keypoints_ragged_tensor = utils.agents_keypoints_to_ragged_tensor(
          agents_in_odometry_df
      )
      tf.data.Dataset.from_tensors(agents_keypoints_ragged_tensor).save(
          os.path.join(output_path, scene, 'agents', 'keypoints')
      )

    robot_in_odometry_df = utils.robot_to_odometry_frame(robot_df)
    robot_pos = tf.convert_to_tensor(
        np.stack(robot_in_odometry_df.iloc[::subsample]['p'].values).astype(
            np.float32
        )
    )
    robot_orientation = tf.convert_to_tensor(
        np.stack(robot_in_odometry_df.iloc[::subsample]['yaw'].values).astype(
            np.float32
        )
    )[..., tf.newaxis]

    tf.data.Dataset.from_tensors(robot_pos).save(
        os.path.join(output_path, scene, 'robot', 'position')
    )
    tf.data.Dataset.from_tensors(robot_orientation).save(
        os.path.join(output_path, scene, 'robot', 'orientation')
    )

    if _PROCESS_POINTCLOUDS.value:
      scene_pointcloud_dict = utils.get_scene_poinclouds(
          os.path.join(input_path, 'train_dataset'),
          scene,
          subsample=subsample,
      )

      # Remove extra timesteps
      scene_pointcloud_dict = {
          ts: scene_pointcloud_dict[ts] for ts in agents_df_subsampled_index
      }

      scene_pc_odometry = utils.pc_to_odometry_frame(
          scene_pointcloud_dict, robot_df
      )

      filtered_pc = utils.filter_agents_and_ground_from_point_cloud(
          agents_in_odometry_df, scene_pc_odometry, robot_in_odometry_df,
          max_dist=_MAX_PC_DISTANCE_TO_ROBOT.value,
      )

      scene_pc_ragged_tensor = tf.ragged.stack(filtered_pc)

      assert (
          agents_pos_ragged_tensor.bounding_shape()[1]
          == scene_pc_ragged_tensor.shape[0]
      )

      tf.data.Dataset.from_tensors(scene_pc_ragged_tensor).save(
          os.path.join(output_path, scene, 'scene', 'pc'), compression='GZIP'
      )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  jrdb_preprocess_train(_INPUT_PATH.value, _OUTPUT_PATH.value)

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'input_path', 'output_path'
  ])
  app.run(main)
