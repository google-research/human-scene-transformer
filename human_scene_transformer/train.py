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

"""Train human scene transformer."""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import gin

from human_scene_transformer import train_model
from human_scene_transformer import training_params as tp
from human_scene_transformer.jrdb import dataset_params as jrdb_dp
from human_scene_transformer.jrdb import input_fn as jrdb_input_fn
from human_scene_transformer.model import model_params as mp
from human_scene_transformer.pedestrians import dataset_params as pedestrians_dp
from human_scene_transformer.pedestrians import input_fn as pedestrians_input_fn

import tensorflow as tf


_DEVICE = flags.DEFINE_string(
    'device', 'cpu', 'The device identification.'
)
_REPLACE_DATE_TIME_STRING_BY = flags.DEFINE_string(
    'replace_date_time_string_by',
    None,
    (
        'A string to replace the automatic date-time string for the model'
        ' foldername.'
    ),
)
_MODEL_BASE_DIR = flags.DEFINE_string(
    'model_base_dir',
    '/tmp/model_dir',
    (
        'The base folder for the trained model, contains ckpts, etc. '
        'Usually this comes from xmanager.'
    ),
)
_TENSORBOARD_DIR = flags.DEFINE_string(
    'tensorboard_dir',
    '/tmp/tensorboard',
    'The base folder to save tensorboard summaries.',
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 1e-4, 'The maximum learning rate'
)
_DATASET = flags.DEFINE_string(
    'dataset', 'EDR', 'Dataset to train and evaluate on.'
)

_GIN_FILE = flags.DEFINE_multi_string(
    'gin_files',
    None,
    'A newline separated list of paths to Gin configuration files used to '
    'configure training, model, and dataset generation')

_GIN_OVERRIDE = flags.DEFINE_multi_string(
    'gin_overrides',
    [],
    'A newline separated list of parameters used to override ones specified in'
    ' the gin file.')

_LOGGING_INTERVAL = 1


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(_GIN_FILE.value, _GIN_OVERRIDE.value)
  logging.info('Actual gin config used:')
  logging.info(gin.config_str())

  if 'cpu' in _DEVICE.value or 'gpu' in _DEVICE.value:
    strategy = tf.distribute.OneDeviceStrategy(_DEVICE.value)
  else:  # TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=_DEVICE.value)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

  train_params = tp.TrainingParams()

  if _DATASET.value == 'JRDB':
    dataset_params = jrdb_dp.JRDBDatasetParams()
    ds_train = jrdb_input_fn.get_train_dataset(dataset_params)
    ds_eval = jrdb_input_fn.get_eval_dataset(dataset_params)
    ds_train = ds_train.batch(train_params.batch_size, drop_remainder=True)
    ds_eval = ds_eval.batch(train_params.batch_size, drop_remainder=True)
    dist_train_dataset = strategy.experimental_distribute_dataset(ds_train)
    dist_eval_dataset = strategy.experimental_distribute_dataset(ds_eval)
  elif _DATASET.value == 'PEDESTRIANS':
    dataset_params = pedestrians_dp.PedestriansDatasetParams()
    ds_train = pedestrians_input_fn.get_train_dataset(dataset_params)
    ds_eval = pedestrians_input_fn.get_eval_dataset(dataset_params)
    ds_train = ds_train.batch(train_params.batch_size, drop_remainder=True)
    ds_eval = ds_eval.batch(train_params.batch_size, drop_remainder=True)
    dist_train_dataset = strategy.experimental_distribute_dataset(ds_train)
    dist_eval_dataset = strategy.experimental_distribute_dataset(ds_eval)
  else:
    raise ValueError

  model_params = mp.ModelParams()

  with strategy.scope():
    metrics_tuple = train_model.get_metrics(model_params)

  train_model.train_model(
      train_params=train_params,
      model_params=model_params,
      train_ds=dist_train_dataset,
      eval_ds=dist_eval_dataset,
      metrics=metrics_tuple,
      strategy=strategy,
      replace_date_time_string_by=_REPLACE_DATE_TIME_STRING_BY.value,
      model_base_dir=_MODEL_BASE_DIR.value,
      tensorboard_dir=_TENSORBOARD_DIR.value
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'model_base_dir', 'gin_files'
  ])
  app.run(main)
