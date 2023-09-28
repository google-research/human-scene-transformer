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

"""Preprocesses the raw test split of JRDB.
"""

import os

from human_scene_transformer.data import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

INPUT_PATH = '<dataset_path>'
OUTPUT_PATH = '<output_path>'

POINTCLOUD = True
AGENT_KEYPOINTS = True
FROM_DETECTIONS = True


def list_test_scenes(input_path):
  scenes = os.listdir(os.path.join(input_path, 'images', 'image_0'))
  scenes.sort()
  return scenes


def get_agents_features_df_with_box(
    input_path, scene_id, max_distance_to_robot=10.0
):
  """Returns agents features with bounding box from raw leaderboard data."""
  jrdb_header = [
      'frame',
      'track id',
      'type',
      'truncated',
      'occluded',
      'alpha',
      'bb_left',
      'bb_top',
      'bb_width',
      'bb_height',
      'x',
      'y',
      'z',
      'height',
      'width',
      'length',
      'rotation_y',
      'score',
  ]
  scene_data_file = utils.get_file_handle(
      os.path.join(
          input_path, 'labels', 'raw_leaderboard', f'{scene_id:04}' + '.txt'
      )
  )
  df = pd.read_csv(scene_data_file, sep=' ', names=jrdb_header)

  def camera_to_lower_velodyne(p):
    return np.stack(
        [p[..., 2], -p[..., 0], -p[..., 1] + (0.742092 - 0.606982)], axis=-1
    )

  df = df[df['score'] >= 0.01]

  df['p'] = df[['x', 'y', 'z']].apply(
      lambda s: camera_to_lower_velodyne(s.to_numpy()), axis=1
  )
  df['distance'] = df['p'].apply(lambda s: np.linalg.norm(s, axis=-1))
  df['l'] = df['height']
  df['h'] = df['width']
  df['w'] = df['length']
  df['yaw'] = df['rotation_y']

  df['id'] = df['track id'].apply(lambda s: f'pedestrian:{s}')
  df['timestep'] = df['frame']

  df = df.set_index(['timestep', 'id'])

  df = df[df['distance'] <= max_distance_to_robot]

  return df[['p', 'yaw', 'l', 'h', 'w']]


def jrdb_preprocess_test(input_path, output_path):
  scenes = list_test_scenes(os.path.join(input_path, 'test_dataset'))
  subsample = 1
  for scene in tqdm.tqdm(scenes):
    scene_save_name = scene + '_test'
    agents_df = get_agents_features_df_with_box(
        os.path.join(input_path, 'test_dataset'),
        scenes.index(scene),
        max_distance_to_robot=15.0,
    )

    robot_odom = utils.get_robot(
        os.path.join(input_path, 'processed', 'odometry_test'), scene
    )

    if AGENT_KEYPOINTS:
      keypoints = utils.get_agents_keypoints(
          os.path.join(
              input_path, 'processed', 'labels', 'labels_3d_keypoints_test'
          ),
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
        os.path.join(output_path, scene_save_name, 'agents', 'position')
    )
    tf.data.Dataset.from_tensors(agents_yaw_ragged_tensor).save(
        os.path.join(output_path, scene_save_name, 'agents', 'orientation')
    )

    if AGENT_KEYPOINTS:
      agents_keypoints_ragged_tensor = utils.agents_keypoints_to_ragged_tensor(
          agents_in_odometry_df
      )
      tf.data.Dataset.from_tensors(agents_keypoints_ragged_tensor).save(
          os.path.join(output_path, scene_save_name, 'agents', 'keypoints')
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
        os.path.join(output_path, scene_save_name, 'robot', 'position')
    )
    tf.data.Dataset.from_tensors(robot_orientation).save(
        os.path.join(output_path, scene_save_name, 'robot', 'orientation')
    )

    if POINTCLOUD:
      scene_pointcloud_dict = utils.get_scene_poinclouds(
          os.path.join(input_path, 'test_dataset'), scene, subsample=subsample
      )
      # Remove extra timesteps
      scene_pointcloud_dict = {
          ts: scene_pointcloud_dict[ts] for ts in agents_df_subsampled_index
      }

      scene_pc_odometry = utils.pc_to_odometry_frame(
          scene_pointcloud_dict, robot_df
      )

      filtered_pc = utils.filter_agents_and_ground_from_point_cloud(
          agents_in_odometry_df, scene_pc_odometry, robot_in_odometry_df
      )

      scene_pc_ragged_tensor = tf.ragged.stack(filtered_pc)

      assert (
          agents_pos_ragged_tensor.bounding_shape()[1]
          == scene_pc_ragged_tensor.shape[0]
      )

      tf.data.Dataset.from_tensors(scene_pc_ragged_tensor).save(
          os.path.join(output_path, scene_save_name, 'scene', 'pc'),
          compression='GZIP',
      )

if __name__ == '__main__':
  jrdb_preprocess_test(INPUT_PATH, OUTPUT_PATH)
