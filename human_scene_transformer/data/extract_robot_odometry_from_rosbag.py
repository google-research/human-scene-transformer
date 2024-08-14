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

"""Extracts robot odometry from JRDB raw rosbags."""


import collections
import glob
import json
import os

from typing import Sequence
from absl import app
from absl import flags
from absl import logging

import pandas as pd
import rosbag


_INPUT_PATH = flags.DEFINE_string(
    'input_path',
    None,
    'Input path to folder with rosbags.',
)

_TIMESTAMPS_PATH = flags.DEFINE_string(
    'timestamps_path',
    None,
    'Input path to folder with timestamps.',
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Output path',
)


def extract_odometry(bag_file):
  """Extracts robot odometry from JRDB raw rosbags."""
  bag = rosbag.Bag(bag_file)

  data_list = list()
  for _, msg, t in bag.read_messages(topics=['segway/feedback/wheel_odometry']):
    data_point = dict()
    data_point['timestamp'] = t.to_sec()
    data_point['pose'] = dict()
    data_point['pose']['position'] = collections.OrderedDict()
    data_point['pose']['orientation'] = collections.OrderedDict()

    data_point['pose']['position']['x'] = msg.pose.pose.position.x
    data_point['pose']['position']['y'] = msg.pose.pose.position.y
    data_point['pose']['position']['z'] = msg.pose.pose.position.z

    data_point['pose']['orientation']['x'] = msg.pose.pose.orientation.x
    data_point['pose']['orientation']['y'] = msg.pose.pose.orientation.y
    data_point['pose']['orientation']['z'] = msg.pose.pose.orientation.z
    data_point['pose']['orientation']['w'] = msg.pose.pose.orientation.w
    data_list.append(data_point)

  return data_list


def fix_timestamps(data_list, timestamp_file):
  """Fix timestamps with expected format by JRDB dataset."""
  with open(timestamp_file) as f:
    timestamps = json.load(f)['data']

  df = pd.DataFrame(data_list).set_index('timestamp')

  data_dict = collections.OrderedDict()
  for timestamp in timestamps:
    ts_id = timestamp['pointclouds'][0]['url'].split('/')[-1]
    ts = timestamp['pointclouds'][0]['timestamp']
    data_dict[ts_id] = df.iloc[df.index.get_indexer([ts], method='nearest')[0]][
        'pose'
    ]

  return data_dict


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  bag_files = glob.glob('*.bag', root_dir=_INPUT_PATH.value)
  for bag_file in bag_files[1:2]:
    data_list = extract_odometry(os.path.join(_INPUT_PATH.value, bag_file))
    fixed_data_list = fix_timestamps(
        data_list,
        os.path.join(_TIMESTAMPS_PATH.value, bag_file[:-4], 'frames_pc.json'),
    )
    data_dict = {'odometry': fixed_data_list}
    with open(
        os.path.join(_OUTPUT_PATH.value, bag_file.replace('bag', 'json')), 'w'
    ) as f:
      json.dump(data_dict, f, indent=2)

if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  app.run(main)
