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

"""box_utils.py holds several useful utilities for working with 7DOF boxes.

Here a 7DOF box is expressed with 7 values representing its center position x,
y, z and shape length, width, height along with the final value for heading in
radians.

The key. functions to use from this library are:
  box3d_to_corners()         -  for converting a 7DOF box into 8x3D corners.
  compute_paired_box3d_iou() -  for computing the intersection over union volume
                                ratio between a pair of 7DOF boxes.

"""
from typing import Optional, Tuple, Union

import tensorflow as tf


def box3d_to_corners(box: tf.Tensor) -> tf.Tensor:
  """Converts a yaw orientated box in 3D to eight corners.

  Args:
    box: Input float tensor with shape [..., 7] where the inner dimensions are
            as follows:[x, y, z, length, width, height, yaw] where x, y, z are
              located at the centre of the box and length, width and height are
              along the unrotated x, y, and z directions respectively. Yaw is in
              radians.

  Returns:
    A tensor with shape [..., 8, 3] representing the eight corners of each box
    provided to the input. Here ... matches the batch rank and shape of the
    input tensor. The corners have the following order:

                 z
                 ^
                 |

             2 --------- 1
            /|          /|
           / |         / |
          3 --------- 0  |
          |  |        |  |    --> y
          |  6 -------|- 5
          | /         | /
          |/          |/
          7 --------- 4


            /
           x       (axis here only indicates direction in unrotated form)

  """

  x, y, z, l, w, h, yaw = tf.split(box, 7, axis=-1)

  x_corners = tf.constant([[1, -1, -1, 1, 1, -1, -1, 1]], box.dtype) * l / 2.
  y_corners = tf.constant([[1, 1, -1, -1, 1, 1, -1, -1]], box.dtype) * w / 2.
  z_corners = tf.constant([[1, 1, 1, 1, -1, -1, -1, -1]], box.dtype) * h / 2.

  # Perform inline rotation for x, y and translation for all x, y, z.
  cos_yaw = tf.cos(yaw)
  sin_yaw = tf.sin(yaw)
  x_rot = x_corners * cos_yaw - y_corners * sin_yaw + x
  y_rot = x_corners * sin_yaw + y_corners * cos_yaw + y
  return tf.stack([x_rot, y_rot, z_corners + z], axis=-1)


def _cross2d(u: tf.Tensor,
             v: tf.Tensor,
             expand: Optional[bool] = False) -> tf.Tensor:
  """Return the cross product scalar of two 2D vectors u and v."""
  cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
  if expand:
    return tf.expand_dims(cross, -1)
  return cross


def _expand_cross2d(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
  return _cross2d(a, b, True)


def _is_point_inside_convex_poly(
    points: tf.Tensor,
    polygon: tf.Tensor,
    include_coincident_points: Optional[bool] = True) -> tf.Tensor:
  """Test if points are inside a convex polygon using cross product method."""

  with tf.name_scope("is_point_inside_convex_poly"):
    pt = tf.expand_dims(points, -2)
    poly = tf.expand_dims(polygon, -3)
    poly_n = tf.roll(poly, 1, -2)

    if include_coincident_points:
      left = tf.greater_equal(_cross2d(poly_n - pt, poly - pt), 0)
    else:
      left = tf.greater(_cross2d(poly_n - pt, poly - pt), 0)
    return tf.reduce_all(left, -1)


def _compute_all_intersection_masks(
    subject_rectangle: tf.Tensor,
    clip_rectangle: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Compute all possible intersections between two rotated rectangles.

  Args:
    subject_rectangle: a float tensor with shape [..., 4, 2] corresponding to
      the four corners of a 2D rotated rectangle.
    clip_rectangle: another float tensor with shape [..., 4, 2] corresponding to
      the four corners of a 2D rotated rectangle.

  Returns:
      A tuple of (intersection_point_matrix, mask_in, mask_out) where:
        intersection_point_matrix - FloatTensor with shape [..., 4, 4, 2],
        mask_in - BoolTensor with shape [..., 4, 4, 1],
        mask_out - BoolTensor with shape [..., 4, 4, 1].
  """

  def _is_p_left_of_uv(u: tf.Tensor, v: tf.Tensor, p: tf.Tensor) -> tf.Tensor:
    return tf.greater(_expand_cross2d(v - u, p - u), 0)

  def _compute_intersection(prev_clip_vertex: tf.Tensor,
                            current_clip_vertex: tf.Tensor,
                            prev_subject_vertex: tf.Tensor,
                            current_subject_vertex: tf.Tensor,
                            eps: Optional[float] = 1e-14) -> tf.Tensor:
    """Returns the intersection of a line segment and an infinite edge."""
    dcp = prev_clip_vertex - current_clip_vertex
    dsp = prev_subject_vertex - current_subject_vertex

    n1 = _expand_cross2d(prev_clip_vertex, current_clip_vertex)
    n2 = _expand_cross2d(prev_subject_vertex, current_subject_vertex)
    divisor = _expand_cross2d(dcp, dsp)
    divisor = tf.where(tf.equal(divisor, 0.), eps, divisor)
    return tf.math.divide(n1 * dsp - n2 * dcp, divisor)

  with tf.name_scope("compute_all_intersections"):
    # prepare segments with broadcastable dimensions
    clip_end = tf.expand_dims(clip_rectangle, -3)
    clip_start = tf.roll(clip_end, 1, -2)
    subject_end = tf.expand_dims(subject_rectangle, -2)
    subject_start = tf.roll(subject_end, 1, -3)

    inside_end = _is_p_left_of_uv(clip_start, clip_end, subject_end)
    inside_start = _is_p_left_of_uv(clip_start, clip_end, subject_start)
    insert_intersection_start = tf.math.logical_xor(inside_end, inside_start)
    inside_end = _is_p_left_of_uv(subject_start, subject_end, clip_end)
    inside_start = _is_p_left_of_uv(subject_start, subject_end, clip_start)
    insert_intersection_c = tf.math.logical_xor(inside_end, inside_start)
    insert_intersection = tf.logical_and(insert_intersection_start,
                                         insert_intersection_c)
    intersection_pts = _compute_intersection(subject_start, subject_end,
                                             clip_start, clip_end)
    intersection_pts_mat = intersection_pts * tf.cast(insert_intersection,
                                                      intersection_pts.dtype)

    mask_in = tf.logical_and(insert_intersection, inside_end)
    mask_out = tf.logical_and(insert_intersection, inside_start)
    return intersection_pts_mat, mask_in, mask_out


def _suppress(mask: tf.Tensor, suppress_mask: tf.Tensor) -> tf.Tensor:
  """Suppress bits in mask where suppress_mask is true."""
  if suppress_mask is not None:
    keep_mask = tf.logical_not(tf.logical_and(mask, suppress_mask))
    mask = tf.logical_and(mask, keep_mask)
  return mask


def intra_seg_area(poly: tf.Tensor,
                   mask: tf.Tensor,
                   suppress_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Compute area of valid segments for a single polygon."""

  with tf.name_scope("intra_seg_area"):
    mask_curr = mask
    mask_prev = tf.roll(mask, 1, 1)
    poly_curr = poly
    poly_prev = tf.roll(poly, 1, 1)
    mask = _suppress(tf.logical_and(mask_curr, mask_prev), suppress_mask)
    mask = tf.cast(mask, poly.dtype)

    return tf.reduce_sum(mask * _cross2d(poly_prev, poly_curr), 1) * .5


def inter_seg_area(inter_pts: tf.Tensor, mask_in: tf.Tensor,
                   mask_out: tf.Tensor, pts0: tf.Tensor, mask0: tf.Tensor,
                   pts1: tf.Tensor, mask1: tf.Tensor) -> tf.Tensor:
  """Compute area from all valid segments using inside + intersection masks."""

  with tf.name_scope("inter_seg_area"):
    roll_dir = 1  # prev
    dtype = inter_pts.dtype

    # fix double counting colinear points by masking in points where intercepted
    seg0intercepted = tf.logical_not(tf.reduce_any(mask_out, [2, 3]))
    mask0 = tf.logical_and(mask0, seg0intercepted)
    seg1intercepted = tf.logical_not(tf.reduce_any(mask_in, [1, 3]))
    mask1 = tf.logical_and(mask1, seg1intercepted)

    in_pts_row = tf.reduce_sum(inter_pts * tf.cast(mask_in, dtype), 2)
    out_pts_row = tf.reduce_sum(inter_pts * tf.cast(mask_out, dtype), 2)
    in_pts_col = tf.reduce_sum(inter_pts * tf.cast(mask_in, dtype), 1)
    out_pts_col = tf.reduce_sum(inter_pts * tf.cast(mask_out, dtype), 1)

    # compute segment masks
    m_out2in_col = tf.logical_and(
        tf.reduce_any(mask_in, 1), tf.reduce_any(mask_out, 1))
    m_in2out_row = tf.logical_and(
        tf.reduce_any(mask_in, 2), tf.reduce_any(mask_out, 2))
    m_out2col = tf.logical_and(tf.reduce_any(mask_out, [1, 3]), mask1)
    m_in2row = tf.logical_and(tf.reduce_any(mask_in, [2, 3]), mask0)
    m_prev_row2out = tf.logical_and(
        tf.reduce_any(mask_out, [2, 3]), tf.roll(mask0, roll_dir, 1))
    m_prev_col2in = tf.logical_and(
        tf.reduce_any(mask_in, [1, 3]), tf.roll(mask1, roll_dir, 1))

    # cast masks into float masks.
    fm_out2col = tf.cast(m_out2col, dtype)
    fm_in2row = tf.cast(m_in2row, dtype)
    fm_prev_row2out = tf.cast(
        _suppress(m_prev_row2out, tf.squeeze(m_in2out_row, -1)), dtype)
    fm_prev_col2in = tf.cast(
        _suppress(m_prev_col2in, tf.squeeze(m_out2in_col, -1)), dtype)
    fm_out2in_col = tf.cast(m_out2in_col, dtype)
    fm_in2out_row = tf.cast(m_in2out_row, dtype)

    # Compute areas of inter-rectangle segments with cross-prod
    # (double the true area)
    a_out2in_col = tf.reduce_sum(
        fm_out2in_col * _expand_cross2d(out_pts_col, in_pts_col), [1, 2])
    a_in2out_row = tf.reduce_sum(
        fm_in2out_row * _expand_cross2d(in_pts_row, out_pts_row), [1, 2])

    a_out2col = tf.reduce_sum(fm_out2col * _cross2d(out_pts_col, pts1), 1)
    a_in2row = tf.reduce_sum(fm_in2row * _cross2d(in_pts_row, pts0), 1)

    a_prev_row2out = tf.reduce_sum(
        fm_prev_row2out * _cross2d(tf.roll(pts0, roll_dir, 1), out_pts_row), 1)
    a_prev_col2in = tf.reduce_sum(
        fm_prev_col2in * _cross2d(tf.roll(pts1, roll_dir, 1), in_pts_col), 1)

    # Gather and sum area for all inter-rectangle segments.
    # Efficiently halve at the end.
    segment_areas = []
    segment_areas.append(a_out2in_col)
    segment_areas.append(a_in2out_row)
    segment_areas.append(a_out2col)
    segment_areas.append(a_in2row)
    segment_areas.append(a_prev_row2out)
    segment_areas.append(a_prev_col2in)
    a_inter_segs_total = tf.add_n(segment_areas) * .5

    # compute area of valid intra-rectangle segments.
    a_rect0 = intra_seg_area(pts0, mask0, m_in2row)
    a_rect1 = intra_seg_area(pts1, mask1, m_out2col)

  # return combined area.
  return a_inter_segs_total + a_rect0 + a_rect1


def compute_box2d_intersection_area(boxes2d0: tf.Tensor,
                                    boxes2d1: tf.Tensor) -> tf.Tensor:
  """Return area of interestion polygon of boxes as corners.

  This method works on the principle that the area of a closed polygon could be
  computed as 0.5 * sum_i( cross(u_i, v_i) ) where u and v denote the start and
  end for each segment of all edges of the polygon. Units of area is the square
  units corresponding to the inputs provided.

  Args:
    boxes2d0: Input float tensor with shape [..., 4, 2] corresponding to the
      four corners of a rotated box.
    boxes2d1: Input float tensor with shape [..., 4, 2] corresponding to the
      four corners of a second rotated box.

  Returns:
    A float tensor with shape [..., 1] representing the area of the
    intersection polygon at the overlap of the two input boxes. Here ...
    matches the batch rank and shape of the inputs.


  Precondition:
    Vertices of inputs must be convex.

  """

  with tf.name_scope("compute_box2d_intersection_area"):
    # move close to origin for numerical stability.
    mean_pt = tf.reduce_mean(tf.concat([boxes2d0, boxes2d1], axis=1), 1, True)
    boxes2d0 -= mean_pt
    boxes2d1 -= mean_pt

    pts, out_mask, in_mask = _compute_all_intersection_masks(boxes2d0, boxes2d1)
    corner0in_mask = _is_point_inside_convex_poly(boxes2d0, boxes2d1)
    corner1in_mask = _is_point_inside_convex_poly(boxes2d1, boxes2d0)

    # special case where all points are coincident.
    all_coincident = tf.logical_and(
        tf.reduce_all(corner0in_mask, -1, True),
        tf.reduce_all(corner1in_mask, -1, True))
    corner1in_mask = tf.logical_and(corner1in_mask,
                                    tf.logical_not(all_coincident))

    intersection_area = inter_seg_area(pts, in_mask, out_mask, boxes2d0,
                                       corner0in_mask, boxes2d1, corner1in_mask)
    return intersection_area


def compute_paired_box3d_iou(
    box0: tf.Tensor,
    box1: tf.Tensor,
    return_tuple: Optional[bool] = False
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
  """Computes IOU between two paired 3D yaw oriented bounding boxes.

  Args:
    box0: Input tensor with shape [B, 7] where the inner dimensions are as
          follows:[x, y, z, length, width, height, yaw].
    box1: Input tensor with shape [B, 7] where the inner dimensions are as
          follows:[x, y, z, length, width, height, yaw]. The input batch shape B
            must match the input shape of box0.
    return_tuple: Boolean flag to indicate if return should be IoU or a tuple of
      (Intersection_volume, Union_volume).

  Returns:
    A tensor matching the input batch shape [B] with values containing IOU
    of 3D volumes from the box pairs. If return_tuple is true then a tuple of
    two tensors are returned representing the intersection and union volumes.

  """

  if len(box0.get_shape()) != 2 or len(box1.get_shape()) != 2:
    raise ValueError("Both box0 and box1 should have shapes as [B, 7].")

  with tf.name_scope("compute_paired_box3d_iou"):
    # use box size to turn into volumes
    vol0 = tf.reduce_prod(box0[..., 3:6], -1)
    vol1 = tf.reduce_prod(box1[..., 3:6], -1)

    corners0 = box3d_to_corners(box0)
    corners1 = box3d_to_corners(box1)

    # compute intersection area in projection onto xy-plane
    poly0 = corners0[..., :4, :2]
    poly1 = corners1[..., :4, :2]

    # Use fast lookuptable method to get intersecting polygon.
    intersection_area_xy = compute_box2d_intersection_area(poly0, poly1)

    # compute intersecting height along z direction
    min_top_z = tf.minimum(corners0[..., 0, 2], corners1[..., 0, 2])
    max_bottom_z = tf.maximum(corners0[..., 4, 2], corners1[..., 4, 2])
    intersection_z = tf.maximum(min_top_z - max_bottom_z, 0.)

    intersection_vol = intersection_area_xy * intersection_z
    union_vol = vol0 + vol1 - intersection_vol

    if return_tuple:
      return intersection_vol, union_vol

    return tf.math.divide_no_nan(intersection_vol, union_vol)


def compute_paired_bev_iou(
    box0: tf.Tensor,
    box1: tf.Tensor,
    return_tuple: Optional[bool] = False
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
  """Computes BEV IOU between two paired 3D yaw oriented bounding boxes.

  Args:
    box0: Input tensor with shape [B, 5] where the inner dimensions are as
          follows:[x, y, length, width, yaw], ignoring the z-axis.
    box1: Input tensor with shape [B, 5] where the inner dimensions are as
          follows:[x, y, length, width, yaw], ignoring the z-axis. The input
            batch shape B must match the input shape of box0.
    return_tuple: Boolean flag to indicate if return should be IoU or a tuple of
      (Intersection_volume, Union_volume).

  Returns:
    A tensor matching the input batch shape [B] with values containing IOU
    of 3D volumes from the box pairs. If return_tuple is true then a tuple of
    two tensors are returned representing the intersection and union volumes.

  """
  if len(box0.get_shape()) != 2 or len(box1.get_shape()) != 2:
    raise ValueError("Both box0 and box1 should have shapes as [B, 5].")

  def _pad_tensor(t_5dof):
    # Add z (zeros) and height (ones) to the tensor t_5dof.
    x, y, l, w, yaw = tf.unstack(t_5dof, axis=-1)
    z = tf.zeros_like(x)
    h = tf.ones_like(x)
    t_7dof = tf.stack([x, y, z, l, w, h, yaw], axis=-1)
    return t_7dof

  with tf.name_scope("compute_paired_box3d_iou"):
    box0 = _pad_tensor(box0)
    box1 = _pad_tensor(box1)

    return compute_paired_box3d_iou(box0, box1, return_tuple)
