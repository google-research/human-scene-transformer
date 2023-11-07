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

"""Evaluates Model on JRDB. Outputs submission files for challenge."""

import functools
import os
from typing import Sequence
from absl import app
from absl import flags
from absl import logging
import gin
from human_scene_transformer.data.math import pose3
from human_scene_transformer.data.math import rotation3
from human_scene_transformer.jrdb import dataset_params as jrdb_dataset_params
from human_scene_transformer.jrdb import input_fn
from human_scene_transformer.model import model as hst_model
from human_scene_transformer.model import model_params
import pandas as pd
import tensorflow as tf
import tqdm

TEST_SCENES = [
    'cubberly-auditorium-2019-04-22_1_test',
    'discovery-walk-2019-02-28_0_test',
    'discovery-walk-2019-02-28_1_test',
    'food-trucks-2019-02-12_0_test',
    'gates-ai-lab-2019-04-17_0_test',
    'gates-basement-elevators-2019-01-17_0_test',
    'gates-foyer-2019-01-17_0_test',
    'gates-to-clark-2019-02-28_0_test',
    'hewlett-class-2019-01-23_0_test',
    'hewlett-class-2019-01-23_1_test',
    'huang-2-2019-01-25_1_test',
    'huang-intersection-2019-01-22_0_test',
    'indoor-coupa-cafe-2019-02-06_0_test',
    'lomita-serra-intersection-2019-01-30_0_test',
    'meyer-green-2019-03-16_1_test',
    'nvidia-aud-2019-01-25_0_test',
    'nvidia-aud-2019-04-18_1_test',
    'nvidia-aud-2019-04-18_2_test',
    'outdoor-coupa-cafe-2019-02-06_0_test',
    'quarry-road-2019-02-28_0_test',
    'serra-street-2019-01-30_0_test',
    'stlc-111-2019-04-19_1_test',
    'stlc-111-2019-04-19_2_test',
    'tressider-2019-03-16_2_test',
    'tressider-2019-04-26_0_test',
    'tressider-2019-04-26_1_test',
    'tressider-2019-04-26_3_test',
]

TIMESTEP_VISIBILITY_THRESHOLD = 6

_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    None,
    help='Path to model directory.',
)

_CKPT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    None,
    help='Path to model checkpoint.',
)

_DATASET_PATH = flags.DEFINE_string(
    'dataset_path',
    None,
    help='Path to model checkpoint.',
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    help='Path to output.',
)


def maybe_makedir(path):
  if not os.path.exists(path):
   os.makedirs(path)


def _scene_last_indices(feature_dict, num_history_steps):
  agents_position = feature_dict['agents/position']
  scene_length = agents_position.bounding_shape(axis=1)

  last_idx = scene_length - num_history_steps - 1
  return tf.range(last_idx, last_idx + 1, dtype=tf.int64)


def load_challenge_dataset(scene, dataset_params):
  """Loads JRDB Dataset with specified scenes."""
  subsample = dataset_params.subsample

  scene_datasets = [None]
  scene_dataset = input_fn._load_scene_with_features(  # pylint: disable=protected-access
      dataset_params.path, scene, dataset_params.features
  )

  start_skip = scene_dataset.map(
      lambda dp: (dp['agents/position'].bounding_shape(axis=1) - 1) % subsample
  )

  challenge_idx = scene_dataset.map(
      lambda dp: dp['agents/position'].bounding_shape(axis=1) - 1
  )

  scene_dataset = tf.data.Dataset.zip((scene_dataset, start_skip))

  scene_dataset_sub = scene_dataset.map(
      functools.partial(input_fn._subsample, step=subsample)  # pylint: disable=protected-access
  ).cache()

  scene_last_idx_dataset = scene_dataset_sub.map(
      functools.partial(
          _scene_last_indices,
          num_history_steps=dataset_params.num_history_steps,
      )
  ).unbatch()

  scene_dataset_sub = tf.data.Dataset.zip(
      (scene_dataset_sub.repeat(), scene_last_idx_dataset)
  )

  scene_dataset_sub = scene_dataset_sub.prefetch(tf.data.experimental.AUTOTUNE)

  scene_datasets[0] = scene_dataset_sub

  # Merge all scenes
  rand_wind_fun = functools.partial(
      input_fn._get_random_window,  # pylint: disable=protected-access
      split_stop=1.0,
      window_length=dataset_params.num_steps,
      target_num_agents=None,
      max_pad_beginning=0,
      random_focus_agent=False,
      min_distance_to_robot=dataset_params.min_distance_to_robot,
  )

  dataset = tf.data.Dataset.from_tensor_slices(scene_datasets).interleave(
      lambda x: x, cycle_length=len(scene_datasets)
  )

  dataset = dataset.map(rand_wind_fun)

  def merge_challenge_idx(dp, challenge_idx):
    dp['scene/challenge_idx'] = challenge_idx
    return dp

  dataset = tf.data.Dataset.zip((dataset, challenge_idx)).map(
      merge_challenge_idx
  )

  dataset = dataset.map(
      functools.partial(
          input_fn._sample_pointcloud,  # pylint: disable=protected-access
          num_pointcloud_points=dataset_params.num_pointcloud_points,
      ),
  )

  dataset = dataset.map(input_fn._map_to_expected_input_format)  # pylint: disable=protected-access

  return dataset


def cv_robot_pred(pos, orient, vel, _, steps):
  """Constant Velocity robot model."""
  future_pos = []
  future_orient = []
  for _ in range(steps):
    pos = pos + vel
    future_pos.append(pos)
    future_orient.append(orient)

  return tf.stack(future_pos, axis=0), tf.stack(future_orient, axis=0)


def ctrv_robot_pred(pos, orient, vel, turning_rate, steps):
  """Constant Turnrate Constant Velocity robot model."""
  future_pos = []
  future_orient = []
  vel = tf.linalg.norm(vel, axis=-1, keepdims=True)
  for _ in range(steps):
    pos = pos + vel * tf.concat(
        [tf.math.cos(orient), tf.math.sin(orient)], axis=-1
    )
    orient = orient + turning_rate
    future_pos.append(pos)
    future_orient.append(orient)

  return tf.stack(future_pos, axis=0), tf.stack(future_orient, axis=0)


def evaluation(checkpoint_path, dataset_path, output_path):
  """Evaluates Model."""

  tf.keras.utils.set_random_seed(111)

  maybe_makedir(output_path)

  d_params = jrdb_dataset_params.JRDBDatasetParams(
      path=dataset_path,
      features=[
          'agents/position',
          'agents/keypoints',
          'robot/position',
          'robot/orientation',
          'scene/pc',
      ],
      num_agents=None,
  )

  model_p = model_params.ModelParams()

  model = hst_model.HumanTrajectorySceneTransformer(model_p)

  model_loaded = False

  for file_i, scene in tqdm.tqdm(enumerate(TEST_SCENES)):
    dataset = load_challenge_dataset(scene, d_params)

    if not model_loaded:
      _, _ = model(next(iter(dataset.batch(1))), training=False)

      checkpoint_mngr = tf.train.Checkpoint(model=model)
      checkpoint_mngr.restore(checkpoint_path).assert_existing_objects_matched()
      logging.info('Restored checkpoint: %s', checkpoint_path)
      model_loaded = True

    input_batch = next(iter(dataset.batch(1)))

    full_pred, _ = model(input_batch, training=False)

    # Get ML prediction
    ml_indices = tf.squeeze(
        tf.math.argmax(full_pred['mixture_logits'], axis=-1)
    )
    pred = full_pred['agents/position'][..., ml_indices, :]

    challenge_idx = input_batch['scene/challenge_idx'][0]

    # Mask agents which are not visible at challenge_idx
    mask = tf.math.logical_not(
        tf.reduce_all(
            tf.math.is_nan(
                tf.reduce_sum(
                    input_batch['agents/position'][
                        :, :, 12 - TIMESTEP_VISIBILITY_THRESHOLD : 12
                    ],
                    axis=-1,
                )
            ),
            axis=-1,
        )
    )

    pred_array = pred[mask][:, 12:].numpy().transpose(1, 0, 2)

    # Construct dataframe
    names = ['timestep', 'id']
    index = pd.MultiIndex.from_product(
        [
            range(challenge_idx + 6, challenge_idx + 13 * 6, 6),
            range(pred_array.shape[1]),
        ],
        names=names,
    )
    df = pd.DataFrame(
        {'x': pred_array[..., 0].flatten(), 'y': pred_array[..., 1].flatten()},
        index=index,
    )

    # Convert into robot coordinate system
    robot_orientation_yaw = input_batch['robot/orientation'][0, 11]
    robot_translation = input_batch['robot/position'][0, 11]
    robot_velocity = (
        input_batch['robot/position'][0, 11]
        - input_batch['robot/position'][0, 10]
    )
    robot_turn_rate = (
        input_batch['robot/orientation'][0, 11]
        - input_batch['robot/orientation'][0, 10]
    )

    robot_future_translation, robot_future_orientation_yaw = cv_robot_pred(
        robot_translation[..., :2],
        robot_orientation_yaw,
        robot_velocity[..., :2],
        robot_turn_rate,
        steps=12,
    )

    robot_future_translation = robot_future_translation.numpy()
    robot_future_orientation_yaw = robot_future_orientation_yaw.numpy()
    df_robot = pd.DataFrame(
        {
            'pos': list(robot_future_translation),
            'orient': list(robot_future_orientation_yaw),
        },
        index=range(challenge_idx + 6, challenge_idx + 13 * 6, 6),
    )

    rotated_dict = {}
    for index, row in df.iterrows():
      ts = index[0]  # pytype: disable=attribute-error disable=unsupported-operands
      robot_trans = df_robot['pos'][ts]
      robot_orient = df_robot['orient'][ts]
      world_pose_odometry = pose3.Pose3(
          rotation3.Rotation3.from_euler_angles(
              rpy_radians=[0.0, 0.0, robot_orient[0]]
          ),
          tf.concat(
              [robot_trans, tf.zeros_like(robot_trans[0:1])], axis=-1
          ).numpy(),
      )
      odometry_pose_world = world_pose_odometry.inverse()

      world_pose_agent = pose3.Pose3(
          rotation3.Rotation3.from_euler_angles(rpy_radians=[0.0, 0.0, 0.0]),
          [row['x'], row['y'], 0.0],
      )

      odometry_pose_agent = odometry_pose_world * world_pose_agent
      rotated_dict[index] = {
          'x': odometry_pose_agent.translation[0],
          'y': odometry_pose_agent.translation[1],
      }

    df = pd.DataFrame.from_dict(rotated_dict, orient='index').rename_axis(
        ['timestep', 'id']
    )

    for i in range(5):
      df.insert(0, f'a{i}', -1)
    for i in range(2):
      df.insert(0, f'b{i}', 0)
    df.insert(0, 'AgentDesc', 'Pedestrian')

    df.to_csv(
        os.path.join(_OUTPUT_PATH.value, f'{file_i:04d}.txt'),
        sep=' ',
        index=True,
        header=False,
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(
      [os.path.join(_MODEL_PATH.value, 'params', 'operative_config.gin')],
      None,
      skip_unknown=True,
  )
  print('Actual gin config used:')
  print(gin.config_str())

  evaluation(_CKPT_PATH.value, _DATASET_PATH.value, _OUTPUT_PATH.value)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'model_path', 'checkpoint_path', 'dataset_path', 'output_path'
  ])
  logging.set_verbosity(logging.ERROR)
  app.run(main)
