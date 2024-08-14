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

"""Utility functions for 2D, 3D, and 4D vectors (python3).

Vectors are represented as a numpy.ndarray.

Utility functions:
  one_hot_vector - Returns a vector of all 0 except for a single 1 value.
  is_vector_normalized - Efficient check for magnitude 1.

Random vector with distribution:
  random_unit_2 - Uniform on unit circle.
  random_unit_3 - Uniform on unit sphere.
  random_unit_4 - Uniform on 4D unit hypersphere.
"""

import math
from typing import Optional, Text, Type, Union

from human_scene_transformer.data.math import math_types

import numpy as np

# ---------------------------------------------------------------------------
# PyType Definitions

# np.random.RandomState will be deprecated in favor of np.random.Generator.
RngType = np.random.RandomState

# ---------------------------------------------------------------------------
# Constants

# ----------------------------------------------------------------------------
# Error messages for exceptions.
VECTOR_COMPONENTS_MESSAGE = 'Vector has incorrect number of components'
VECTOR_INFINITE_VALUES_MESSAGE = 'Vector has non-finite values'
VECTOR_VALUES_MESSAGE = 'Vector has NaN values'
VECTOR_ZERO_MESSAGE = 'Vector has nearly zero magnitude.'

# ----------------------------------------------------------------------------
# Default values.

# Default error tolerance for the magnitude of a normalized vector.
DEFAULT_NORM_EPSILON = math_types.DEFAULT_RTOL_VALUE_FOR_NP_IS_CLOSE
# Default error tolerance for the magnitude of a normalizable vector.
DEFAULT_ZERO_EPSILON = math_types.DEFAULT_ATOL_VALUE_FOR_NP_IS_CLOSE


def normalize_vector(vector: math_types.VectorType,
                     err_msg: Text = '') -> np.ndarray:
  """Returns a vector in the same direction but with magnitude 1.

  Args:
    vector:
    err_msg:

  Returns:
    A vector in the same direction as the argument but with magnitude 1.

  Raises:
    ValueError: If the vector cannot be normalized.
  """
  vector = np.array(vector, dtype=np.float64)
  vector_norm = np.linalg.norm(vector)
  if vector_norm <= DEFAULT_ZERO_EPSILON:
    raise ValueError('%s: |%r| = %f\n%s' %
                     (VECTOR_ZERO_MESSAGE, vector, vector_norm, err_msg))
  return vector / vector_norm


def one_hot_vector(
    dimension: int,
    hot_index: int,
    dtype: Union[np.dtype, Type[np.number]] = np.float64,
) -> np.ndarray:
  """Returns a vector with all zeros except the hot_index component.

  For example,
    one_hot_vector(3, 0) = [1, 0, 0]
    one_hot_vector(4, 2) = [0, 0, 1, 0]

  Args:
    dimension: Number of components in the vector.
    hot_index: Index of element with value 1.0.
    dtype: Numeric type of components.

  Returns:
    The one-hot vector as a numpy array.
  """
  vector = np.zeros(dimension, dtype=dtype)
  vector[hot_index] = 1
  return vector


def as_vector(
    values: math_types.VectorType,
    dimension: Optional[int] = None,
    dtype: Optional[Union[np.dtype, Type[np.number]]] = None,
    err_msg: Text = '',
) -> np.ndarray:
  """Interprets the values as a vector with <dimension> components.

  All values may be infinite.  NaN values will result in an error.

  Args:
    values: Input vector values.
    dimension: Expected dimension of output vector and number of components in
      input vector.
    dtype: Numeric type of array.
    err_msg: Error message string appended to exception in case of failure.

  Returns:
    The input vector as a numpy array after checking its size and values.

  Raises:
    ValueError: If the inputs are not a valid vector with the correct number of
      components.
  """
  if dimension is not None and len(values) != dimension:
    raise ValueError(
        '%s: Expected %d vector components, but found %d: %r\n%s' %
        (VECTOR_COMPONENTS_MESSAGE, dimension, len(values), values, err_msg))
  vector = np.asarray(values, dtype=dtype)
  if np.any(np.isnan(vector)):
    raise ValueError('%s: %r\n%s' % (VECTOR_VALUES_MESSAGE, values, err_msg))
  return vector


def as_finite_vector(
    values: math_types.VectorType,
    dimension: Optional[int] = None,
    normalize: bool = False,
    dtype: Optional[Union[np.dtype, Type[np.number]]] = None,
    err_msg: Text = '',
) -> np.ndarray:
  """Interprets the values as a vector with <dimension> components.

  All values must be finite.  Infinite or NaN values will result in an error.

  Args:
    values: Input vector values.
    dimension: Expected dimension of output vector and number of components in
      input vector.
    normalize: Indicates whether to normalize the vector.
    dtype: Numeric type of array.
    err_msg: Error message string appended to exception in case of failure.

  Returns:
    The input vector as a numpy array after checking its size and values.

  Raises:
    ValueError: If the inputs are not a valid vector with the correct number of
      components or if normalize is True and the vector has near zero magnitude.
  """
  vector = as_vector(
      values=values, dimension=dimension, dtype=dtype, err_msg=err_msg)
  if not np.all(np.isfinite(vector)):
    raise ValueError('%s: %r\n%s' %
                     (VECTOR_INFINITE_VALUES_MESSAGE, values, err_msg))
  if normalize:
    vector = normalize_vector(vector, err_msg=err_msg)
  return vector


def as_vector2(
    values: math_types.VectorType,
    dtype: Union[np.dtype, Type[np.number]] = np.float64,
    err_msg='',
):
  """Gets input as a finite vector with dimension=2, dtype=float64."""
  return as_finite_vector(values, dimension=2, dtype=dtype, err_msg=err_msg)


def as_unit_vector2(
    values: math_types.VectorType,
    dtype: Union[np.dtype, Type[np.number]] = np.float64,
    err_msg='',
):
  """Gets input as a finite vector with dimension=2, dtype=float64, normalize=True."""
  return as_finite_vector(
      values, dimension=2, normalize=True, dtype=dtype, err_msg=err_msg)


def as_vector3(
    values: math_types.VectorType,
    dtype: Union[np.dtype, Type[np.number]] = np.float64,
    err_msg='',
):
  """Gets input as a finite vector with dimension=3, dtype=float64."""
  return as_finite_vector(values, dimension=3, dtype=dtype, err_msg=err_msg)


def as_unit_vector3(
    values: math_types.VectorType,
    dtype: Union[np.dtype, Type[np.number]] = np.float64,
    err_msg='',
):
  """Gets input as a finite vector with dimension=3, dtype=float64, normalize=True."""
  return as_finite_vector(
      values, dimension=3, normalize=True, dtype=dtype, err_msg=err_msg)


def as_vector4(
    values: math_types.VectorType,
    dtype: Union[np.dtype, Type[np.number]] = np.float64,
    err_msg='',
):
  """Gets input as a finite vector with dimension=4, dtype=float64."""
  return as_finite_vector(values, dimension=4, dtype=dtype, err_msg=err_msg)


def as_unit_vector4(
    values: math_types.VectorType,
    dtype: Union[np.dtype, Type[np.number]] = np.float64,
    err_msg='',
):
  """Gets input as a finite vector with dimension=4, dtype=float64, normalize=True."""
  return as_finite_vector(
      values, dimension=4, normalize=True, dtype=dtype, err_msg=err_msg)


def is_vector_normalized(vector: math_types.VectorType,
                         norm_epsilon: float = DEFAULT_NORM_EPSILON) -> bool:
  """Returns true if |vector| = 1 within epsilon tolerance.

  If norm_epsilon is zero, only vectors that have magnitude precisely 1.0 will
  be considered normalized.

  This is an efficient computation without sqrt, but it is only accurate within
  norm_epsilon squared.

  Args:
    vector: Vector to check for magnitude = 1.
    norm_epsilon: Error tolerance on magnitude of vector.

  Returns:
    True if the magnitude of the vector is within epsilon of 1.0.
  """
  vector = np.asarray(vector)
  v_dot_v = vector.dot(vector)
  # The error in the squared norm varies linearly with error in component
  # values when the vector is close to unit magnitude.
  #
  # If |v| = 1 + x with x << 1
  #   then v.v = (1 + x)^2 = 1 + 2x + x^2
  #   and |1.0 - v.v| = |2x + x^2|
  #   which is close to 2|x|
  return abs(1 - v_dot_v) <= norm_epsilon * 2


def default_rng(seed: Optional[int] = None) -> RngType:
  """Returns a random number generator with the given seed."""
  return RngType(seed=seed)


def random_unit_2(dtype=np.float64,
                  rng: Optional[RngType] = None) -> np.ndarray:
  """Returns a random 2D vector with unit length.

  Generates a 2D vector selected uniformly from the unit circle.

  Args:
    dtype: Numeric type of result.
    rng: A random number generator.

  Returns:
    A normalized 2D vector in a numpy array.
  """
  if not rng:
    rng = np.random
  angle = rng.uniform(low=-math.pi, high=math.pi)
  x = math.cos(angle)
  y = math.sin(angle)
  return np.array([x, y], dtype=dtype)


def random_unit_3(dtype=np.float64,
                  rng: Optional[RngType] = None) -> np.ndarray:
  """Returns a random 3D vector with unit length.

  Generates a 3D vector selected uniformly from the unit sphere.

  Args:
    dtype: Numeric type of result.
    rng: A random number generator.

  Returns:
    A normalized 3D vector in a numpy array.
  """
  if not rng:
    rng = np.random
  longitude = rng.uniform(low=-math.pi, high=math.pi)
  sin_latitude = rng.uniform(low=-1.0, high=1.0)
  cos_latitude = math.sqrt(1.0 - sin_latitude**2)
  x = math.cos(longitude) * cos_latitude
  y = math.sin(longitude) * cos_latitude
  z = sin_latitude
  return np.array([x, y, z], dtype=dtype)


def random_unit_4(dtype=np.float64,
                  rng: Optional[RngType] = None) -> np.ndarray:
  """Returns a random 4D vector with unit length.

  Generates a 4D vector selected uniformly from the unit hypersphere.
  Can be used to generate uniform random quaternions for rotation.

  Args:
    dtype: Numeric type of result.
    rng: A random number generator.

  Returns:
    A normalized 4D vector in a numpy array.
  """
  if not rng:
    rng = np.random
  u1 = rng.uniform(low=0.0, high=1.0)
  u2 = rng.uniform(low=-math.pi, high=math.pi)
  u3 = rng.uniform(low=-math.pi, high=math.pi)
  x = math.sqrt(1 - u1) * math.cos(u2)
  y = math.sqrt(1 - u1) * math.sin(u2)
  z = math.sqrt(u1) * math.cos(u3)
  w = math.sqrt(u1) * math.sin(u3)
  return np.array([x, y, z, w], dtype=dtype)
