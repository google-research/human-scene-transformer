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

"""Evaluates Model on JRDB dataset."""

import os
from typing import Sequence
from absl import app
from absl import flags
from absl import logging
import gin
from human_scene_transformer.jrdb import dataset_params
from human_scene_transformer.jrdb import input_fn
from human_scene_transformer.metrics import metrics
from human_scene_transformer.model import model as hst_model
from human_scene_transformer.model import model_params
import tensorflow as tf
import tqdm


_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    None,
    'Path to model directory.',
)

_CKPT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    None,
    'Path to model checkpoint.',
)


def filter_non_moving_agents(input_dict, output_dict):
  """Filters non-moving agents."""
  pos = input_dict['agents/position'][..., :2]
  dist = tf.linalg.norm(
      tf.reduce_min(tf.where(tf.math.is_nan(pos), tf.float32.max, pos), axis=-2)
      - tf.reduce_max(
          tf.where(tf.math.is_nan(pos), tf.float32.min, pos), axis=-2
      ),
      axis=-1,
      keepdims=True,
  )[..., tf.newaxis]
  output_dict['should_predict'] = tf.math.logical_and(
      output_dict['should_predict'], dist > 0.5
  )

  return output_dict


def filter_agents_without_keypoints(input_dict, output_dict):
  """Filters agents without keypoints."""
  num_key = tf.reduce_sum(
      tf.cast(
          tf.logical_not(
              tf.reduce_any(
                  tf.math.is_nan(
                      input_dict['agents/keypoints'][:, :, :7]
                      ), axis=-1, keepdims=True)
              ), tf.float32), axis=-2, keepdims=True)

  output_dict['should_predict'] = tf.math.logical_and(
      output_dict['should_predict'], num_key > 5)

  return output_dict


def evaluation(checkpoint_path):
  """Evaluates Model on Pedestrian dataset."""
  d_params = dataset_params.JRDBDatasetParams(num_agents=None)

  dataset = input_fn.load_dataset(
      d_params,
      d_params.eval_scenes,
      augment=False,
      shuffle=False,
      allow_parallel=False,
      evaluation=False,
      repeat=False,
      keep_subsamples=False,
  )

  model_p = model_params.ModelParams()

  model = hst_model.HumanTrajectorySceneTransformer(model_p)

  _, _ = model(next(iter(dataset.batch(1))), training=False)

  checkpoint_mngr = tf.train.Checkpoint(model=model)
  checkpoint_mngr.restore(checkpoint_path).assert_existing_objects_matched()
  logging.info('Restored checkpoint: %s', checkpoint_path)

  ade_metric = metrics.ade.MinADE(model_p)
  ade_metric_1s = metrics.ade.MinADE(
      model_p, cutoff_seconds=1.0, at_cutoff=True
  )
  ade_metric_2s = metrics.ade.MinADE(
      model_p, cutoff_seconds=2.0, at_cutoff=True
  )
  ade_metric_3s = metrics.ade.MinADE(
      model_p, cutoff_seconds=3.0, at_cutoff=True
  )
  ade_metric_4s = metrics.ade.MinADE(
      model_p, cutoff_seconds=4.0, at_cutoff=True
  )

  mlade_metric = metrics.ade.MLADE(model_p)
  mlade_metric_1s = metrics.ade.MLADE(
      model_p, cutoff_seconds=1.0, at_cutoff=True
  )
  mlade_metric_2s = metrics.ade.MLADE(
      model_p, cutoff_seconds=2.0, at_cutoff=True
  )
  mlade_metric_3s = metrics.ade.MLADE(
      model_p, cutoff_seconds=3.0, at_cutoff=True
  )
  mlade_metric_4s = metrics.ade.MLADE(
      model_p, cutoff_seconds=4.0, at_cutoff=True
  )

  nll_metric = metrics.pos_nll.PositionNegativeLogLikelihood(model_p)
  nll_metric_1s = metrics.pos_nll.PositionNegativeLogLikelihood(
      model_p, cutoff_seconds=1.0, at_cutoff=True
  )
  nll_metric_2s = metrics.pos_nll.PositionNegativeLogLikelihood(
      model_p, cutoff_seconds=2.0, at_cutoff=True
  )
  nll_metric_3s = metrics.pos_nll.PositionNegativeLogLikelihood(
      model_p, cutoff_seconds=3.0, at_cutoff=True
  )
  nll_metric_4s = metrics.pos_nll.PositionNegativeLogLikelihood(
      model_p, cutoff_seconds=4.0, at_cutoff=True
  )

  for input_batch in tqdm.tqdm(dataset.batch(1)):
    full_pred, output_batch = model(input_batch, training=False)
    output_batch = filter_agents_without_keypoints(input_batch, output_batch)
    output_batch = filter_non_moving_agents(input_batch, output_batch)
    ade_metric.update_state(output_batch, full_pred)
    ade_metric_1s.update_state(output_batch, full_pred)
    ade_metric_2s.update_state(output_batch, full_pred)
    ade_metric_3s.update_state(output_batch, full_pred)
    ade_metric_4s.update_state(output_batch, full_pred)

    mlade_metric.update_state(output_batch, full_pred)
    mlade_metric_1s.update_state(output_batch, full_pred)
    mlade_metric_2s.update_state(output_batch, full_pred)
    mlade_metric_3s.update_state(output_batch, full_pred)
    mlade_metric_4s.update_state(output_batch, full_pred)

    nll_metric.update_state(output_batch, full_pred)
    nll_metric_1s.update_state(output_batch, full_pred)
    nll_metric_2s.update_state(output_batch, full_pred)
    nll_metric_3s.update_state(output_batch, full_pred)
    nll_metric_4s.update_state(output_batch, full_pred)

  print(f'MinADE: {ade_metric.result().numpy():.2f}')
  print(f'MinADE @ 1s: {ade_metric_1s.result().numpy():.2f}')
  print(f'MinADE @ 2s: {ade_metric_2s.result().numpy():.2f}')
  print(f'MinADE @ 3s: {ade_metric_3s.result().numpy():.2f}')
  print(f'MinADE @ 4s: {ade_metric_4s.result().numpy():.2f}')

  print(f'MLADE: {mlade_metric.result().numpy():.2f}')
  print(f'MLADE @ 1s: {mlade_metric_1s.result().numpy():.2f}')
  print(f'MLADE @ 2s: {mlade_metric_2s.result().numpy():.2f}')
  print(f'MLADE @ 3s: {mlade_metric_3s.result().numpy():.2f}')
  print(f'MLADE @ 4s: {mlade_metric_4s.result().numpy():.2f}')

  print(f'NLL: {nll_metric.result().numpy():.2f}')
  print(f'NLL @ 1s: {nll_metric_1s.result().numpy():.2f}')
  print(f'NLL @ 2s: {nll_metric_2s.result().numpy():.2f}')
  print(f'NLL @ 3s: {nll_metric_3s.result().numpy():.2f}')
  print(f'NLL @ 4s: {nll_metric_4s.result().numpy():.2f}')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(
      [os.path.join(_MODEL_PATH.value, 'params', 'operative_config.gin')],
      None,
      skip_unknown=True)
  print('Actual gin config used:')
  print(gin.config_str())

  evaluation(_CKPT_PATH.value)

if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  app.run(main)
