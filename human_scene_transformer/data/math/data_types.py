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

"""A hidden layer to define all of the type definitions."""

from typing import Any, Optional, Sequence

from human_scene_transformer.data.math import math_types
from human_scene_transformer.data.math import pose3
from human_scene_transformer.data.math import quaternion
from human_scene_transformer.data.math import rotation3
from human_scene_transformer.data.math import vector_util

import numpy as np


VECTOR_TYPE = math_types.VectorType
Rotation3 = rotation3.Rotation3
Pose3 = pose3.Pose3
Quaternion = quaternion.Quaternion


class Twist:
  """A class which represents a twist with linear and angular components."""

  def __init__(self,
               linear: Optional[VECTOR_TYPE] = None,
               angular: Optional[VECTOR_TYPE] = None):
    """Constructs a new twist.

    Initializes a new twist object.

    If no parameters are specified the default will be a zero twist.

    Args:
      linear: Linear component of twist, zero by default.
      angular: Angular component of twist, zero by default.
    """
    if linear is None:
      self.linear = np.zeros(3)
    else:
      self.linear = vector_util.as_vector3(
          linear, dtype=np.float64, err_msg='Linear vector in Twist').copy()
    if angular is None:
      self.angular = np.zeros(3)
    else:
      self.angular = vector_util.as_vector3(
          angular, dtype=np.float64, err_msg='Angular vector in Twist').copy()

  @property
  def vec(self) -> np.ndarray:
    """Returns six-value representation: [linear, angular]."""
    return np.hstack((self.linear, self.angular))

  def __repr__(self) -> str:
    """Generates a string that describes the twist.

    Returns:
      A string describing the twist.
    """
    return 'Twist({}, {})'.format(self.linear, self.angular)


class Wrench:
  """A class which represents a wrench with force and torque components."""

  def __init__(self,
               force: Optional[VECTOR_TYPE] = None,
               torque: Optional[VECTOR_TYPE] = None):
    """Constructs a new wrench.

    Initializes a new wrench object.

    If no parameters are specified the default will be a zero wrench.

    Args:
      force: Force, zero by default.
      torque: Torque, zero by default.
    """
    if force is None:
      self.force = np.zeros(3)
    else:
      self.force = vector_util.as_vector3(
          force, dtype=np.float64, err_msg='Force vector in Wrench').copy()
    if torque is None:
      self.torque = np.zeros(3)
    else:
      self.torque = vector_util.as_vector3(
          torque, dtype=np.float64, err_msg='Torque vector in Wrench').copy()

  @property
  def vec(self) -> np.ndarray:
    """Returns six-value representation: [force, torque]."""
    return np.hstack((self.force, self.torque))

  def __repr__(self) -> str:
    """Generates a string that describes the wrench.

    Returns:
      A string describing the wrench.
    """
    return 'Wrench({}, {})'.format(self.force, self.torque)


class Stiffness:
  """A class which represents stiffness."""

  def __init__(self,
               linear: Optional[VECTOR_TYPE] = None,
               torsional: Optional[VECTOR_TYPE] = None,
               matrix6x6: Optional[np.ndarray] = None):
    """Constructs a new stiffness.

    Initializes a new stiffness object.

    If no parameters are specified the default will be a zero stiffness.

    Args:
      linear: Linear component of stiffness, zero by default.
      torsional: Torsional component of stiffness, zero by default.
      matrix6x6: A full 6x6 stiffness matrix.  When 'matrix6x6' is provided,
        'linear' and 'torsional' are ignored and the linear and torsional
        components of Stiffness will be set as the diagonal elements of
        'matrix6x6'.

    Raises:
      ValueError if the shape of 'matrix6x6' is not 6x6.
    """
    if matrix6x6 is not None:
      if not np.array_equal(matrix6x6.shape, [6, 6]):
        raise ValueError('Invalid matrix6x6 shape: {0}'.format(matrix6x6.shape))
      self.matrix6x6 = np.copy(matrix6x6).astype(np.float64)
      diag = self.matrix6x6.diagonal()
      self.linear = diag[:3]
      self.torsional = diag[3:]
    else:
      # Sets linear stiffness.
      if linear is None:
        self.linear = np.zeros(3)
      else:
        self.linear = vector_util.as_vector3(
            linear, dtype=np.float64,
            err_msg='Linear vector in Stiffness').copy()
      # Sets torsional stiffness.
      if torsional is None:
        self.torsional = np.zeros(3)
      else:
        self.torsional = vector_util.as_vector3(
            torsional, dtype=np.float64,
            err_msg='Angular vector in Stiffness').copy()
      # Sets the 6x6 stiffness matrix.
      self.matrix6x6 = np.diag(self.vec)

  @classmethod
  def from_scalar(cls, scalar: float) -> 'Stiffness':
    return cls(linear=[scalar] * 3, torsional=[scalar] * 3)

  @property
  def vec(self) -> np.ndarray:
    """Returns the diagonal elements: [linear, torsional]."""
    return np.hstack((self.linear, self.torsional))

  def __repr__(self) -> str:
    """Generates a string that describes the stiffness.

    Returns:
      A string describing the stiffness.
    """
    return 'Stiffness({})'.format(self.matrix6x6)


# Convenience functions.
def vec6_to_pose3(vec6: VECTOR_TYPE) -> Pose3:
  """Creates a Pose3 object from a given 6-D vector.

  Args:
    vec6: A 6-D vector, [x, y, z, rx, ry, rz], that denotes translation [x, y,
      z] and orientation [rx, ry, rz] as Euler angles roll, pitch, yaw, in
      radians.

  Returns:
    A Pose3 object.
  """
  assert len(vec6) == 6
  return Pose3(
      translation=vec6[:3],
      rotation=Rotation3.from_euler_angles(rpy_radians=vec6[3:]))


def pose3_to_vec6(pose: Pose3) -> np.ndarray:
  """Returns a 6-D vector representation of the pose.

  Args:
    pose: A Pose3 object.

  Returns:
    A 6-D vector, [x, y, z, rx, ry, rz], that denotes translation
    [x, y, z] and orientation [rx, ry, rz].
  """
  return np.hstack((pose.translation, pose.rotation.euler_angles()))


def pose3_to_vec6_in_axis_angles(pose: Pose3) -> np.ndarray:
  """Returns a 6-D vector representation of the pose.

  Args:
    pose: A Pose3 object.

  Returns:
    A 6-D vector, [x, y, z, rx, ry, rz], that denotes translation
    [x, y, z] and orientation [rx, ry, rz] as axis angles.
  """
  return np.hstack(
      (pose.translation, pose.rotation.axis() * pose.rotation.angle()))


def vec6_to_twist(vec6: VECTOR_TYPE) -> Twist:
  """Creates a Twist object from a given 6-D vector.

  Args:
    vec6: A 6-D vector, [vx, vy, vz, wx, wy, wz], that denotes linear velocity
      along X, Y, Z and angular velocity around X, Y, Z.

  Returns:
    A Twist object.
  """
  assert len(vec6) == 6
  return Twist(linear=vec6[:3], angular=vec6[3:])


def vec6_to_wrench(vec6: VECTOR_TYPE) -> Wrench:
  """Creates a Wrench object from a given 6-D vector.

  Args:
    vec6: A 6-D vector, [fx, fy, fz, tx, ty, tz], that denotes force and torque.

  Returns:
    A Wrench object.
  """
  assert len(vec6) == 6
  return Wrench(force=vec6[:3], torque=vec6[3:])


def vec6_to_stiffness(vec6: VECTOR_TYPE) -> Stiffness:
  """Creates a Stiffness object from a given 6-D vector.

  Args:
    vec6: A 6-D vector, [x, y, z, rx, ry, rz], that denotes translational and
      rotational stiffness for X, Y, Z.

  Returns:
    A Stiffness object.
  """
  assert len(vec6) == 6
  return Stiffness(linear=vec6[:3], torsional=vec6[3:])


def to_list(data: Any) -> Sequence[float]:
  """Flattens a common data type to a list of float.

  Args:
    data: The input data to flatten.

  Returns:
    The flattend list, i.e., the vectorized parametrization of the input data.
  """
  if isinstance(data, float):
    return [data]
  elif isinstance(data, np.ndarray):
    return data.tolist()
  elif isinstance(data, Pose3):
    return data.vec7.tolist()
  elif isinstance(data, Twist):
    return data.vec.tolist()
  elif isinstance(data, Wrench):
    return data.vec.tolist()
  else:
    raise ValueError('{} is not supported in to_list()'.format(type(data)))
