from __future__ import print_function

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

def flip_boxes_left_right(boxes):
    """Left-right flip the boxes.

    Args:
        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
             Boxes are in normalized form meaning their coordinates vary
             between [0, 1].
             Each row is in the form of [ymin, xmin, ymax, xmax].

    Returns:
        Flipped boxes.
    """
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    flipped_xmin = tf.subtract(1.0, xmax)
    flipped_xmax = tf.subtract(1.0, xmin)
    flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
    return flipped_boxes

def prune_completely_outside_window(boxes, window, scope=None):
    """Prunes bounding boxes that fall completely outside of the given window.

    The function clip_to_window prunes bounding boxes that fall
    completely outside the window, but also clips any bounding boxes that
    partially overflow. This function does not clip partially overflowing boxes.

    """
    with tf.name_scope(scope, 'PruneCompleteleyOutsideWindow'):
        y_min, x_min, y_max, x_max = tf.split(
                value=boxes, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        coordinate_violations = tf.concat([
                tf.greater_equal(y_min, win_y_max), tf.greater_equal(x_min, win_x_max),
                tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
        ], 1)
        valid_indices = tf.reshape(
                tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
        return tf.gather(boxes, valid_indices), valid_indices

def prune_non_overlapping_boxes(
        boxes1, boxes2, min_overlap=0.0, scope=None):
    """Prunes the boxes in boxes1 that overlap less than thresh with boxes2.

    For each box in boxes1, we want its IOA to be more than minoverlap with
    at least one of the boxes in boxes2. If it does not, we remove it.

    Args:
        boxes1: BoxList holding N boxes.
        boxes2: BoxList holding M boxes.
        min_overlap: Minimum required overlap between boxes, to count them as
                                overlapping.
        scope: name scope.

    Returns:
        new_boxes1: A pruned boxlist with size [N', 4].
        keep_inds: A tensor with shape [N'] indexing kept bounding boxes in the
            first input BoxList `boxes1`.
    """
    with tf.name_scope(scope, 'PruneNonOverlappingBoxes'):
        ioa_ = ioa(boxes2, boxes1)    # [M, N] tensor
        ioa_ = tf.reduce_max(ioa_, reduction_indices=[0])    # [N] tensor
        keep_bool = tf.greater_equal(ioa_, tf.constant(min_overlap))
        keep_inds = tf.squeeze(tf.where(keep_bool), squeeze_dims=[1])
        new_boxes1 = tf.gather(boxes1, keep_inds)
        return new_boxes1, keep_inds

def ioa(boxes1, boxes2, scope=None):
    """Computes pairwise intersection-over-area between box collections.

    intersection-over-area (IOA) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, ioa(box1, box2) != ioa(box2, box1).

    Args:
        boxes1: BoxList holding N boxes
        boxes2: BoxList holding M boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise ioa scores.
    """
    with tf.name_scope(scope, 'IOA'):
        intersections = intersection(boxes1, boxes2)
        areas = tf.expand_dims(area(boxes2), 0)
        return tf.truediv(intersections, areas)

def intersection(boxes1, boxes2, scope=None):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: BoxList holding N boxes
        boxes2: BoxList holding M boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'Intersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
                value=boxes1, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
                value=boxes2, num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths

def area(boxes, scope=None):
    """Computes area of boxes.

    Args:
        boxlist: BoxList holding N boxes
        scope: name scope.

    Returns:
        a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'Area'):
        y_min, x_min, y_max, x_max = tf.split(
                value=boxes, num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

def change_coordinate_frame(boxes, window, scope=None):
    """Change coordinate frame of the boxlist to be relative to window's frame.

    Given a window of the form [ymin, xmin, ymax, xmax],
    changes bounding box coordinates from boxlist to be relative to this window
    (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

    An example use case is data augmentation: where we are given groundtruth
    boxes (boxlist) and would like to randomly crop the image to some
    window (window). In this case we need to change the coordinate frame of
    each groundtruth box to be relative to this new window.

    Args:
        boxlist: A BoxList object holding N boxes.
        window: A rank 1 tensor [4].
        scope: name scope.

    Returns:
        Returns a BoxList object with N boxes.
    """
    with tf.name_scope(scope, 'ChangeCoordinateFrame'):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        boxes_new = scale(boxes - [window[0], window[1], window[0], window[1]],
                                                1.0 / win_height, 1.0 / win_width)
        return boxes_new

def scale(boxes, y_scale, x_scale, scope=None):
    """scale box coordinates in x and y dimensions.

    Args:
        boxlist: BoxList holding N boxes
        y_scale: (float) scalar tensor
        x_scale: (float) scalar tensor
        scope: name scope.

    Returns:
        boxlist: BoxList holding N boxes
    """
    with tf.name_scope(scope, 'Scale'):
        y_scale = tf.cast(y_scale, tf.float32)
        x_scale = tf.cast(x_scale, tf.float32)
        y_min, x_min, y_max, x_max = tf.split(
                value=boxes, num_or_size_splits=4, axis=1)
        y_min = y_scale * y_min
        y_max = y_scale * y_max
        x_min = x_scale * x_min
        x_max = x_scale * x_max
        scaled_boxes = tf.concat([y_min, x_min, y_max, x_max], 1)
        return scaled_boxes

def iou(boxes1, boxes2, scope=None):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: BoxList holding N boxes
        boxes2: BoxList holding M boxes
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'IOU'):
        intersections = intersection(boxes1, boxes2)
        areas1 = area(boxes1)
        areas2 = area(boxes2)
        unions = (
                tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
        return tf.where(
                tf.equal(intersections, 0.0),
                tf.zeros_like(intersections), tf.truediv(intersections, unions))

def get_center_coordinates_and_sizes(boxes, scope=None):
    """Computes the center coordinates, height and width of the boxes.

    Args:
      scope: name scope of the function.

    Returns:
      a list of 4 1-D tensors [ycenter, xcenter, height, width].
    """
    with tf.name_scope(scope, 'get_center_coordinates_and_sizes'):
        ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(boxes))
        width = xmax - xmin
        height = ymax - ymin
        ycenter = ymin + height / 2.
        xcenter = xmin + width / 2.
        return [ycenter, xcenter, height, width]

def matched_iou(boxes1, boxes2, scope=None):
    """Compute intersection-over-union between corresponding boxes in boxlists.

    Args:
        boxes1: BoxList holding N boxes
        boxes2: BoxList holding N boxes
        scope: name scope.

    Returns:
        a tensor with shape [N] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'MatchedIOU'):
        intersections = matched_intersection(boxes1, boxes2)
        areas1 = area(boxes1)
        areas2 = area(boxes2)
        unions = areas1 + areas2 - intersections
        return tf.where(
                tf.equal(intersections, 0.0),
                tf.zeros_like(intersections), tf.truediv(intersections, unions))

