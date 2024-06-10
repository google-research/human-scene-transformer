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

"""Fuses detections to tracks.
"""


import json
import os

from absl import app
from absl import flags

from human_scene_transformer.data import box_utils
from human_scene_transformer.data import utils
import numpy as np
import pandas as pd
import scipy
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


def get_agents_3d_bounding_box_dict(input_path, scene):
  """Returns a dict of agent labels and their bounding boxes."""
  scene_data_file = utils.get_file_handle(
      os.path.join(input_path, 'labels', 'labels_3d', scene + '.json')
  )
  scene_data = json.load(scene_data_file)

  agents = {}

  for frame in scene_data['labels']:
    ts = int(frame.split('.')[0])
    for det in scene_data['labels'][frame]:
      agents[(ts, det['label_id'])] = {
          'box': np.array([
              det['box']['cx'],
              det['box']['cy'],
              det['box']['l'],
              det['box']['w'],
              det['box']['rot_z']]),
          'box3d': np.array([
              det['box']['cx'],
              det['box']['cy'],
              det['box']['cz'],
              det['box']['l'],
              det['box']['w'],
              det['box']['h'],
              det['box']['rot_z']])
          }
  return agents


def get_agents_3d_bounding_box_detections_dict(input_path, scene):
  """Returns a dict of agent detections and their bounding boxes."""
  scene_data_file = utils.get_file_handle(
      os.path.join(input_path, 'detections', 'detections_3d', scene + '.json')
  )

  scene_data = json.load(scene_data_file)

  agents = []

  for frame in scene_data['detections']:
    ts = int(frame.split('.')[0])
    for det in scene_data['detections'][frame]:
      agents.append((
          ts,
          np.array(
              [det['box']['cx'],
               det['box']['cy'],
               det['box']['l'],
               det['box']['w'],
               det['box']['rot_z']]),
          np.array([
              det['box']['cx'],
              det['box']['cy'],
              # Detections should be in sensor coordinate
              # frame (0.6m from ground).
              det['box']['cz'] - 0.606982,
              det['box']['l'],
              det['box']['w'],
              det['box']['h'],
              det['box']['rot_z']]),
          det['score']
      ))
  return agents


def detections_to_dict(df):
  """Puts a detections dataframe into expected dict format."""
  labels_dict = {}
  for ts, group in df.groupby('timestep'):
    agent_list = []
    for (_, agent_id), row in group.iterrows():
      agent_list.append({'label_id': agent_id,
                         'box': {
                             'cx': row['detection'][0],
                             'cy': row['detection'][1],
                             'cz': row['detection'][2],
                             'l': row['detection'][3],
                             'w': row['detection'][4],
                             'h': row['detection'][5],
                             'rot_z': row['detection'][6],
                         },
                         'attributes': {
                             'distance': np.linalg.norm(row['detection'][:3])
                             }})
    labels_dict[f'{ts:06}.pcb'] = agent_list
  return {'labels': labels_dict}


def jrdb_train_detections_to_tracks(input_path, output_path):
  """Fuses detections to tracks."""
  utils.maybe_makedir(output_path)
  scenes = utils.list_scenes(
      os.path.join(input_path, 'train_dataset'))
  for scene in tqdm.tqdm(scenes):
    bb_dict = get_agents_3d_bounding_box_dict(
        os.path.join(input_path, 'train_dataset'), scene)
    bb_3d_df = pd.DataFrame.from_dict(
        bb_dict, orient='index').rename_axis(['timestep', 'id'])  # pytype: disable=missing-parameter  # pandas-drop-duplicates-overloads

    bb_detections_list = get_agents_3d_bounding_box_detections_dict(
        os.path.join(input_path, 'train_dataset'), scene)
    bb_3d_detections_df = pd.DataFrame(
        bb_detections_list, columns=['timestep', 'box', 'box3d', 'score'])

    dfs = []
    for ts, gt_group in bb_3d_df.groupby('timestep'):
      detections_group = bb_3d_detections_df.loc[
          (bb_3d_detections_df['timestep'] == ts)]
      detection_boxes = np.vstack(detections_group['box'])
      gt_boxes = np.vstack(gt_group['box'])

      detection_boxes_rep = np.repeat(
          detection_boxes, gt_boxes.shape[0], axis=0)
      gt_boxes_til = np.tile(gt_boxes, (detection_boxes.shape[0], 1))

      iou = box_utils.compute_paired_bev_iou(
          tf.convert_to_tensor(detection_boxes_rep),
          tf.convert_to_tensor(gt_boxes_til)
          )
      assert isinstance(iou, tf.Tensor)
      iou_np = iou.numpy().reshape(
          (detection_boxes.shape[0], gt_boxes.shape[0]))
      cost = 1 - iou_np
      r, c = scipy.optimize.linear_sum_assignment(cost.T)

      unmatched_gt = np.argwhere(cost.T[r, c] == 1.)[..., 0]

      df_tmp = gt_group.iloc[r].copy()
      df_tmp['detection'] = detections_group['box3d'].iloc[c].to_list()

      df_tmp = df_tmp.drop(index=df_tmp.index[unmatched_gt])

      dfs.append(df_tmp)

    matched_df = pd.concat(dfs)

    labels_dict = detections_to_dict(matched_df)

    with open(f"{output_path}/{scene}.json", 'w') as write_file:
      json.dump(labels_dict, write_file, indent=2, ensure_ascii=True)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if _OUTPUT_PATH.value is None:
    output_path = os.path.join(_INPUT_PATH.value,
                               'processed/labels/labels_detections_3d')
  else:
    output_path = _OUTPUT_PATH.value
  jrdb_train_detections_to_tracks(_INPUT_PATH.value, output_path)

if __name__ == '__main__':
  flags.mark_flags_as_required(['input_path'])
  app.run(main)
