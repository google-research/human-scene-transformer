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

"""Preprocesses the raw test split of JRDB.
"""

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

_TRACKING_METHOD = flags.DEFINE_string(
    'tracking_method',
    default='ss3d_mot',
    help='Name of tracking method to use.'
)

_TRACKING_CONFIDENCE_THRESHOLD = flags.DEFINE_float(
    'tracking_confidence_threshold',
    default=.0,
    help=('Confidence threshold for tracked agent instance to be included'
          ' in the processed dataset.')
)

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
          input_path, 'labels', _TRACKING_METHOD.value,
          f'{scene_id:04}' + '.txt'
      )
  )
  df = pd.read_csv(scene_data_file, sep=' ', names=jrdb_header)

  def camera_to_lower_velodyne(p):
    return np.stack(
        [p[..., 2], -p[..., 0], -p[..., 1] + (0.742092 - 0.606982)], axis=-1
    )

  df = df[df['score'] >= _TRACKING_CONFIDENCE_THRESHOLD.value]

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
  """Preprocesses the raw test split of JRDB."""

  tf.keras.utils.set_random_seed(123)

  scenes = list_test_scenes(os.path.join(input_path, 'test_dataset'))
  subsample = 1
  for scene in tqdm.tqdm(scenes):
    scene_save_name = scene + '_test'
    agents_df = get_agents_features_df_with_box(
        os.path.join(input_path, 'test_dataset'),
        scenes.index(scene),
        max_distance_to_robot=_MAX_DISTANCE_TO_ROBOT.value,
    )

    robot_odom = utils.get_robot(
        os.path.join(input_path, 'processed', 'odometry', 'test'), scene
    )

    if AGENT_KEYPOINTS:
      keypoints = utils.get_agents_keypoints(
          os.path.join(
              input_path, 'processed', 'labels',
              'labels_3d_keypoints', 'test', _TRACKING_METHOD.value
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

    if _PROCESS_POINTCLOUDS.value:
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
          agents_in_odometry_df, scene_pc_odometry, robot_in_odometry_df,
          max_dist=_MAX_PC_DISTANCE_TO_ROBOT.value,
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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  jrdb_preprocess_test(_INPUT_PATH.value, _OUTPUT_PATH.value)

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'input_path', 'output_path'
  ])
  app.run(main)
