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

"""Train function."""

import datetime
import os
from typing import Text, Tuple

from absl import logging
import gin

from human_scene_transformer import training_params as tp
from human_scene_transformer.metrics import metrics as hst_metrics
from human_scene_transformer.model import model as hst_model
from human_scene_transformer.model import model_params as mp

import tensorflow as tf

import official.modeling.optimization.lr_schedule as tfm_lr_schedule

_LOGGING_INTERVAL = 1


def get_file_handle(path, mode='rt'):
  file_handle = open(path, mode)
  return file_handle


def make_dirs(path):
  os.makedirs(path)


@gin.configurable()
def get_metrics(model_params, train_metrics=None, eval_metrics=None):
  """Create train and eval metrics. Designed for gin configurable."""

  metrics_list = [train_metrics, eval_metrics]
  # Instantiate metrics.
  for m in metrics_list:
    for key in m:
      # Standard Keras metric.
      if m[key] == hst_metrics.Mean:
        m[key] = m[key](dtype=tf.float32)
      else:
        m[key] = m[key](model_params)
  return train_metrics, eval_metrics


def _get_learning_rate_schedule(
    warmup_steps: int,
    total_steps: int,
    learning_rate: float,
    alpha: float = 0.0) -> tf.keras.optimizers.schedules.LearningRateSchedule:
  """Returns a cosine decay learning rate schedule to be used in training.

  Args:
    warmup_steps: Number of training steps to apply warmup. If global_step <
      warmup_steps, the learning rate will be `global_step/num_warmup_steps *
      init_lr`.
    total_steps: The total number of training steps.
    learning_rate: The peak learning rate anytime during the training.
    alpha: The alpha parameter forwarded to CosineDecay

  Returns:
    A CosineDecay learning schedule w/ warmup.
  """

  decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
      initial_learning_rate=learning_rate, decay_steps=total_steps, alpha=alpha)
  return tfm_lr_schedule.LinearWarmup(decay_schedule, warmup_steps, 1e-10)


def train_model(
    train_params: tp.TrainingParams,
    model_params: mp.ModelParams,
    train_ds: tf.data.Dataset,
    eval_ds: tf.data.Dataset,
    metrics: Tuple[tf.keras.metrics.Metric, tf.keras.metrics.Metric],
    strategy: tf.distribute.Strategy,
    replace_date_time_string_by: Text,
    model_base_dir: Text,
    tensorboard_dir: Text,
) -> tf.keras.Model:
  """Runs training and evaluation.

  Args:
    train_params: A TrainingParams object that contains parameters required for
      model training.
    model_params: A ModelParams object that conifgures the model architecture.
    train_ds: Training Dataset.
    eval_ds: Evaluation Dataset.
    metrics: Tuple of Train and Eval Metrics.
    strategy: A distribution strategy to run training under.
    replace_date_time_string_by: A String which replaces the date added by
      default.
    model_base_dir: The directory in which model checkpoints are saved.
    tensorboard_dir: The base folder to save tensorboard summaries.

  Returns:
    model: A trained Keras model.
  """

  # 1) Create folders and gin config file used for training.
  if replace_date_time_string_by is None:
    dt_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
  else:
    dt_str = replace_date_time_string_by
  model_dir = os.path.join(model_base_dir, dt_str)
  make_dirs(model_dir)
  ckpt_dir = os.path.join(model_dir, 'ckpts')
  make_dirs(ckpt_dir)
  ckpt_best_dir = os.path.join(model_dir, 'ckpts_best')
  make_dirs(ckpt_best_dir)
  checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')
  checkpoint_prefix_best = os.path.join(ckpt_best_dir, 'ckpt')

  param_prefix = os.path.join(model_dir, 'params')
  make_dirs(param_prefix)
  gin_operative_config = gin.operative_config_str()
  with get_file_handle(
      os.path.join(param_prefix, 'operative_config.gin'), 'w') as f:
    f.write(gin_operative_config)

  # 2) Create cosine learning rate schedule w/ warmup.
  learning_rate_schedule = _get_learning_rate_schedule(
      train_params.warmup_steps, train_params.total_train_steps,
      train_params.peak_learning_rate)

  # 3) Create model and metrics on Device.
  with strategy.scope():
    model = hst_model.HumanTrajectorySceneTransformer(params=model_params)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule,
        global_clipnorm=train_params.global_clipnorm)
    loss_obj = train_params.loss(model_params)

    # Create metrics.
    train_metrics, eval_metrics = metrics
  logging.info('Model created on device.')

  best_eval_loss = tf.Variable(tf.float32.max)

  checkpoint = tf.train.Checkpoint(model=model,
                                   optimizer=optimizer,
                                   best_eval_loss=best_eval_loss)
  latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
  current_global_step = 0
  if latest_checkpoint:
    checkpoint.restore(latest_checkpoint).assert_existing_objects_matched()
    logging.info('Restored from checkpoint: %s', latest_checkpoint)
    current_global_step = optimizer.iterations.numpy()

  checkpoint_best = tf.train.Checkpoint(model=model)
  best_checkpoint_manager = tf.train.CheckpointManager(checkpoint_best,
                                                       checkpoint_prefix_best,
                                                       max_to_keep=1)

  # 4) Create summary writers.
  train_summary_writer = tf.summary.create_file_writer(
      os.path.join(tensorboard_dir, 'train'))
  eval_summary_writer = tf.summary.create_file_writer(
      os.path.join(tensorboard_dir, 'eval'))

  # Define Training and Eval tf.function.
  @tf.function
  def train_step(iterator):
    """Training function."""

    def step_fn(input_batch):
      with tf.GradientTape() as tape:
        predictions, output_batch = model(input_batch, training=True)
        loss_dict = loss_obj(output_batch, predictions)
        loss = (loss_dict['loss']
                / tf.cast(strategy.num_replicas_in_sync, tf.float32))

      grads = tape.gradient(loss, model.trainable_variables)

      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      # Update the training metrics.
      # These need special treatments as they are standard keras metrics.
      train_metrics['loss'].update_state(loss_dict['loss'])
      train_metrics['loss_position'].update_state(loss_dict['position_loss'])
      if 'orientation_loss' in loss_dict.keys():
        train_metrics['loss_orientation'].update_state(
            loss_dict['orientation_loss'])
      # Our own metrics.
      for key in train_metrics:
        if key in {'loss', 'loss_position', 'loss_orientation'}:
          continue
        train_metrics[key].update_state(output_batch, predictions)

    for _ in tf.range(tf.constant(train_params.batches_per_train_step)):
      strategy.run(
          step_fn,
          args=(next(iterator),),
          options=tf.distribute.RunOptions(
              experimental_enable_dynamic_batch_size=False))

  @tf.function
  def eval_step(iterator):

    def step_fn(input_batch):
      predictions, output_batch = model(input_batch, training=False)
      loss_dict = loss_obj(output_batch, predictions)
      # Update the eval metrics.
      # These need special treatments as they are standard keras metrics.
      eval_metrics['loss'].update_state(loss_dict['loss'])
      eval_metrics['loss_position'].update_state(loss_dict['position_loss'])
      if 'orientation_loss' in loss_dict.keys():
        eval_metrics['loss_orientation'].update_state(
            loss_dict['orientation_loss'])
      # Our own metrics.
      for key in eval_metrics:
        if key in {'loss', 'loss_position', 'loss_orientation'}:
          continue
        eval_metrics[key].update_state(output_batch, predictions)

    for _ in tf.range(tf.constant(train_params.batches_per_eval_step)):
      strategy.run(
          step_fn,
          args=(next(iterator),),
          options=tf.distribute.RunOptions(
              experimental_enable_dynamic_batch_size=False))

  # 5) Actual Training Loop
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)
  num_train_iter = (
      train_params.total_train_steps // train_params.batches_per_train_step)
  current_train_iter = (
      current_global_step // train_params.batches_per_train_step)

  logging.info('Beginning training.')
  for step in range(current_train_iter, num_train_iter):
    # Actual number of SGD steps.
    actual_steps = step * train_params.batches_per_train_step
    with train_summary_writer.as_default():
      # Run training SGD over train_param.batches_per_train_step batches.
      # optimizer.iterations = step * train_param.batches_per_train_step.
      train_step(train_iter)
      # Writing metrics to tensorboard.
      if step % _LOGGING_INTERVAL == 0:
        for key in train_metrics:
          tf.summary.scalar(
              key, train_metrics[key].result(), step=optimizer.iterations)

        if isinstance(optimizer, tf.keras.optimizers.experimental.Optimizer):
          learning_rate = optimizer.learning_rate
        else:
          learning_rate = optimizer.lr(optimizer.iterations)
        tf.summary.scalar(
            'learning_rate',
            learning_rate,
            step=optimizer.iterations)
        logging.info('Training step %d', step)
        logging.info('Training loss: %.4f, ADE: %.4f',
                     train_metrics['loss'].result().numpy(),
                     train_metrics['min_ade'].result().numpy())
        # Reset metrics.
        for key in train_metrics:
          train_metrics[key].reset_states()

    # Evaluation.
    if actual_steps % train_params.eval_every_n_step == 0:
      logging.info('Evaluating step %d over %d random eval samples', step,
                   train_params.batches_per_eval_step * train_params.batch_size)
      with eval_summary_writer.as_default():
        eval_step(eval_iter)
        for key in eval_metrics:
          tf.summary.scalar(
              key, eval_metrics[key].result(), step=optimizer.iterations)
        logging.info('Eval loss: %.4f, ADE: %.4f',
                     eval_metrics['loss'].result().numpy(),
                     eval_metrics['min_ade'].result().numpy())

        if eval_metrics['loss'].result() < best_eval_loss:
          best_eval_loss.assign(eval_metrics['loss'].result())
          best_checkpoint_manager.save()

        # Reset metrics.
        for key in eval_metrics:
          eval_metrics[key].reset_states()

        # Save model.
        checkpoint_name = checkpoint.save(checkpoint_prefix)
        logging.info('Saved checkpoint to %s', checkpoint_name)

  return model
