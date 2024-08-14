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

"""Utility Functions."""

import collections
import json
import os
import tempfile

from human_scene_transformer.data.math import pose3
from human_scene_transformer.data.math import quaternion
from human_scene_transformer.data.math import rotation3

import numpy as np
import pandas as pd

import tensorflow as tf

import open3d


def maybe_makedir(path):
  if not os.path.exists(path):
   os.makedirs(path)


def get_file_handle(path, mode='rt'):
  file_handle = open(path, mode)
  return file_handle


def list_scenes(input_path):
  scenes = os.listdir(os.path.join(input_path, 'labels', 'labels_3d'))
  scenes.sort()
  return [scene[:-5] for scene in scenes]


def get_robot(input_path, scene):
  """Returns robot features from raw data."""
  odom_data_file = get_file_handle(
      os.path.join(input_path, scene + '.json'))
  odom_data = json.load(odom_data_file)

  robot = collections.defaultdict(list)

  for pc_ts, pose in odom_data['odometry'].items():
    ts = int(pc_ts.split('.')[0])
    robot[ts] = {
        'p': np.array([pose['position']['x'],
                       pose['position']['y'],
                       pose['position']['z']]),
        'q': np.array([pose['orientation']['x'],
                       pose['orientation']['y'],
                       pose['orientation']['z'],
                       pose['orientation']['w']]),
    }

  return robot


def get_agents_keypoints(input_path, scene):
  """Returns agents keypoints from raw data."""
  scene_data_file = get_file_handle(
      os.path.join(input_path, scene + '.json'))
  scene_data = json.load(scene_data_file)

  agents_keypoints = collections.defaultdict(dict)

  for frame in scene_data['labels']:
    ts = int(frame.split('.')[0])
    for det in scene_data['labels'][frame]:
      agents_keypoints[(ts, det['label_id'])] = {
          'keypoints': np.array(det['keypoints']).reshape(33, 3)}
  return agents_keypoints


def get_scene_poinclouds(input_path, scene, subsample=1):
  """Returns scene point clouds from raw data."""
  pc_files = os.listdir(
      os.path.join(input_path, 'pointclouds', 'lower_velodyne', scene))

  pc_files = sorted(pc_files)
  pc_dict = collections.OrderedDict()
  for _, pc_file in enumerate(pc_files[::subsample]):
    pc_file_path = os.path.join(
        input_path, 'pointclouds', 'lower_velodyne', scene, pc_file)
    with get_file_handle(pc_file_path, 'rb') as f:
      with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(f.read())
        pcd = open3d.io.read_point_cloud(tmp.name, format='pcd')
        pc_dict[int(pc_file.split('.')[0])] = pcd
  return pc_dict


def pc_to_odometry_frame(pc_dict, robot_df):
  """Transforms point clouds into odometry frame."""
  world_pose_odometry = pose3.Pose3(
      rotation3.Rotation3(
          quaternion.Quaternion(robot_df.loc[0]['q'])), robot_df.loc[0]['p'])
  odometry_pose_world = world_pose_odometry.inverse()

  pc_list = []
  for ts, pc in pc_dict.items():
    robot_odometry_dp = robot_df.loc[ts]

    world_pose_robot = pose3.Pose3(
        rotation3.Rotation3(
            quaternion.Quaternion(
                robot_odometry_dp['q'])), robot_odometry_dp['p'])

    odometry_pose_robot = odometry_pose_world * world_pose_robot

    odometry_pc = pc.transform(odometry_pose_robot.matrix4x4())

    pc_list.append(np.array(odometry_pc.points, dtype=np.float32))

  return pc_list


def robot_to_odometry_frame(robot_df):
  """Transforms robot features into odometry frame."""
  world_pose_odometry = pose3.Pose3(
      rotation3.Rotation3(
          quaternion.Quaternion(robot_df.loc[0]['q'])), robot_df.loc[0]['p'])
  odometry_pose_world = world_pose_odometry.inverse()

  robot_dict = {}
  for ts, row in robot_df.iterrows():
    world_pose_robot = pose3.Pose3(
        rotation3.Rotation3(quaternion.Quaternion(row['q'])), row['p'])
    odometry_pose_robot = odometry_pose_world * world_pose_robot

    robot_dict[ts] = {
        'p': odometry_pose_robot.translation,
        'yaw': odometry_pose_robot.rotation.euler_angles(radians=True)[-1]
        }
  return pd.DataFrame.from_dict(
      robot_dict, orient='index').rename_axis(['timestep'])  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads


def agents_to_odometry_frame(agents_df, robot_df):
  """Transforms agents features into odometry frame."""
  world_pose_odometry = pose3.Pose3(
      rotation3.Rotation3(
          quaternion.Quaternion(robot_df.loc[0]['q'])), robot_df.loc[0]['p'])
  odometry_pose_world = world_pose_odometry.inverse()

  agents_dict = {}
  for index, row in agents_df.iterrows():
    ts = index[0]
    robot_odometry_dp = robot_df.loc[ts]

    world_pose_robot = pose3.Pose3(
        rotation3.Rotation3(
            quaternion.Quaternion(robot_odometry_dp['q'])),
        robot_odometry_dp['p'])

    robot_pose_agent = pose3.Pose3(
        rotation3.Rotation3.from_euler_angles(
            rpy_radians=[0., 0., row['yaw']]), row['p'])

    odometry_pose_agent = (odometry_pose_world * world_pose_robot
                           * robot_pose_agent)

    agents_dict[index] = {
        'p': odometry_pose_agent.translation,
        'yaw': odometry_pose_agent.rotation.euler_angles(radians=True)[-1]}

    if 'l' in row:
      agents_dict[index]['l'] = row['l']
      agents_dict[index]['w'] = row['w']
      agents_dict[index]['h'] = row['h']

    if 'keypoints' in row:
      world_rot_robot = rotation3.Rotation3(
          quaternion.Quaternion(robot_odometry_dp['q']))
      odometry_rot_robot = odometry_pose_world.rotation * world_rot_robot
      rot_keypoints = []
      for keypoint in row['keypoints']:
        if np.isnan(keypoint).any():
          rot_keypoints.append(keypoint)
        else:
          rot_keypoints.append(odometry_rot_robot.rotate_point(keypoint))
      rot_keypoints = np.array(rot_keypoints)
      agents_dict[index]['keypoints'] = rot_keypoints

  return pd.DataFrame.from_dict(
      agents_dict, orient='index').rename_axis(['timestep', 'id'])  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads


def get_agents_features_with_box(agents_dict, max_distance_to_robot=10):
  """Returns agents features with bounding box from raw data dict."""
  agents_pos_dict = collections.defaultdict(dict)
  for agent_id, agent_data in agents_dict.items():
    for (ts, agent_instance) in agent_data:
      if agent_instance['attributes']['distance'] <= max_distance_to_robot:
        agents_pos_dict[(ts, agent_id)] = {
            'p': np.array([agent_instance['box']['cx'],
                           agent_instance['box']['cy'],
                           agent_instance['box']['cz']]),
            # rotation angle is relative to negatiev x axis of robot
            'yaw': np.pi - agent_instance['box']['rot_z'],
            'l': agent_instance['box']['l'],
            'w': agent_instance['box']['w'],
            'h': agent_instance['box']['h']
        }
  return agents_pos_dict


def agents_pos_to_ragged_tensor(agents_df):
  tensor_list = []
  for _, df in agents_df.groupby('id'):
    dropped_df = df.droplevel(1, axis=0)
    r_tensor = tf.RaggedTensor.from_value_rowids(
        values=np.vstack(df['p'].values).flatten().astype(np.float32),
        value_rowids=np.tile(
            np.array(dropped_df.index), (3, 1)).transpose().flatten())
    tensor_list.append(r_tensor)
  return tf.stack(tensor_list)


def agents_yaw_to_ragged_tensor(agents_df):
  tensor_list = []
  for _, df in agents_df.groupby('id'):
    dropped_df = df.droplevel(1, axis=0)
    r_tensor = tf.RaggedTensor.from_value_rowids(
        values=np.vstack(df['yaw'].values).flatten().astype(np.float32),
        value_rowids=np.array(dropped_df.index))
    tensor_list.append(r_tensor)
  return tf.stack(tensor_list)


def agents_keypoints_to_ragged_tensor(agents_df):
  tensor_list = []
  for _, df in agents_df.groupby('id'):
    dropped_df = df.droplevel(1, axis=0)
    r_tensor = tf.RaggedTensor.from_value_rowids(
        values=np.vstack(df['keypoints'].values).astype(np.float32),
        value_rowids=np.tile(
            np.array(dropped_df.index), (33, 1)).transpose().flatten())
    tensor_list.append(r_tensor)
  return tf.stack(tensor_list)


def box_to_hyperplanes(pos, yaw, l, w, h):
  """Transforms a bounding box to a set of hyperplanes."""
  s = np.sin(yaw)
  c = np.cos(yaw)
  normal = np.array([
      np.array([0, 0, h/2]),
      np.array([0, 0, -h/2]),
      np.array([-s * w/2, c * w/2, 0]),
      np.array([-s * -w/2, c * -w/2, 0]),
      np.array([c * l/2, s * l/2, 0]),
      np.array([c * -l/2, s * -l/2, 0])])
  points = pos + normal
  normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
  w = -normal

  d = -w[:, 0] * points[:, 0] - w[:, 1] * points[:, 1] - w[:, 2] * points[:, 2]

  return w, d


def filter_agents_and_ground_from_point_cloud(
    agents_df, pointcloud_dict, robot_in_odometry_df, max_dist=10.):
  """Filter points which are in human bb or belong to ground."""
  for t, agent_df in agents_df.groupby('timestep'):
    pc_points = pointcloud_dict[t]
    robot_p = robot_in_odometry_df.loc[t]['p'][:2]
    dist_mask = np.linalg.norm(robot_p - pc_points[..., :2], axis=-1) < max_dist
    pc_points = pc_points[
        (pc_points[:, -1] > -0.2) & (pc_points[:, -1] < 0.5) & dist_mask]
    for _, row in agent_df.iterrows():
      w, d = box_to_hyperplanes(
          row['p'], row['yaw'], 1.5*row['l'], 1.5*row['w'], row['h'])
      agent_pc_mask = np.all((pc_points @ w.T + d) > 0., axis=-1)
      pc_points = pc_points[~agent_pc_mask]
    np.random.shuffle(pc_points)
    pointcloud_dict[t] = pc_points
  return pointcloud_dict

