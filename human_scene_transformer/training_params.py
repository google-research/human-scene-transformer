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

"""Training parameter class.

The module contains a dataclasses used to configure the training procedure.
"""

import gin

from human_scene_transformer import losses


@gin.configurable
class TrainingParams(object):
  """This object configures the model and training/eval."""

  def __init__(
      self,
      batch_size=32,
      shuffle_buffer_size=10000,
      total_train_steps=5e6,
      warmup_steps=5e4,
      peak_learning_rate=1e-4,
      global_clipnorm=None,
      batches_per_train_step=2000,
      batches_per_eval_step=2000,
      eval_every_n_step=1e4,
      loss=losses.MultimodalPositionOrientationNLLLoss,
  ):
    """Initialize the TrainParam object.

    This is where gin injection occurs.

    Args:
      batch_size: The batch size of training and eval. On TPUs, each core will
        receive a batch size that equals batch_size/num_cores.
      shuffle_buffer_size: The size of the tf.dataset.shuffle() operation.
      total_train_steps: The total number of training steps (will automatically
        be converted to int).
      warmup_steps: The number of warmup steps during training. The learning
        rate will increase linearly to the peak at this step (from 1/10 of
        peak). Will automatically be converted to int.
      peak_learning_rate: The maximum learning rate during training.
      global_clipnorm: Gradient norm clipping value.
      batches_per_train_step: Number of batches per training step. This is
        required to improve the efficiency for Keras custom training loops. Will
        automatically be converted to int.
      batches_per_eval_step: Number of batches per eval step. The total number
        of eval examples will be batches_per_eval_step * batch_size. Will
        automatically be converted to int.
      eval_every_n_step: Number of training steps before evaluating using the
        eval set. Will automatically be converted to int.
      loss: A class used as loss.
    """
    self.batch_size = batch_size
    self.shuffle_buffer_size = shuffle_buffer_size
    self.total_train_steps = int(total_train_steps)
    self.warmup_steps = int(warmup_steps)
    self.peak_learning_rate = peak_learning_rate
    self.global_clipnorm = global_clipnorm
    self.batches_per_train_step = int(batches_per_train_step)
    self.batches_per_eval_step = int(batches_per_eval_step)
    self.eval_every_n_step = int(eval_every_n_step)
    self.loss = loss
