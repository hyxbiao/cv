#!/usr/bin/env python
# encoding: utf-8

import os
import sys

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.utils import ops

class AnchorGenerator(object):

    def generate(self):
        raise NotImplementedError

    def tile_anchors(self,
                    grid_height,
                    grid_width,
                    scales,
                    aspect_ratios,
                    base_anchor_size,
                    anchor_stride,
                    anchor_offset):
        """Create a tiled set of anchors strided along a grid in image space.

        This op creates a set of anchor boxes by placing a "basis" collection of
        boxes with user-specified scales and aspect ratios centered at evenly
        distributed points along a grid.    The basis collection is specified via the
        scale and aspect_ratios arguments.    For example, setting scales=[.1, .2, .2]
        and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
        .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
        and aspect ratio 1/2.    Each box is multiplied by "base_anchor_size" before
        placing it over its respective center.

        Grid points are specified via grid_height, grid_width parameters as well as
        the anchor_stride and anchor_offset parameters.

        Args:
            grid_height: size of the grid in the y direction (int or int scalar tensor)
            grid_width: size of the grid in the x direction (int or int scalar tensor)
            scales: a 1-d    (float) tensor representing the scale of each box in the
                basis set.
            aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each
                box in the basis set.    The length of the scales and aspect_ratios tensors
                must be equal.
            base_anchor_size: base anchor size as [height, width]
                (float tensor of shape [2])
            anchor_stride: difference in centers between base anchors for adjacent grid
                                         positions (float tensor of shape [2])
            anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                                         upper left element of the grid, this should be zero for
                                         feature networks with only VALID padding and even receptive
                                         field size, but may need some additional calculation if other
                                         padding is used (float tensor of shape [2])
        Returns:
            a BoxList holding a collection of N anchor boxes
        """
        ratio_sqrts = tf.sqrt(aspect_ratios)
        heights = scales / ratio_sqrts * base_anchor_size[0]
        widths = scales * ratio_sqrts * base_anchor_size[1]

        # Get a grid of box centers
        y_centers = tf.to_float(tf.range(grid_height))
        y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
        x_centers = tf.to_float(tf.range(grid_width))
        x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
        x_centers, y_centers = ops.meshgrid(x_centers, y_centers)

        widths_grid, x_centers_grid = ops.meshgrid(widths, x_centers)
        heights_grid, y_centers_grid = ops.meshgrid(heights, y_centers)
        bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
        bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
        bbox_centers = tf.reshape(bbox_centers, [-1, 2])
        bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
        bbox_corners = self._center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
        return bbox_corners


    def _center_size_bbox_to_corners_bbox(self, centers, sizes):
        """Converts bbox center-size representation to corners representation.

        Args:
            centers: a tensor with shape [N, 2] representing bounding box centers
            sizes: a tensor with shape [N, 2] representing bounding boxes

        Returns:
            corners: tensor with shape [N, 4] representing bounding boxes in corners
                representation
        """
        return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)


class MultipleGridAnchorGenerator(AnchorGenerator):

    def __init__(self, box_specs_list, base_anchor_size):
        super(MultipleGridAnchorGenerator, self).__init__()
        self._box_specs = box_specs_list
        self._base_anchor_size = base_anchor_size

        self._scales = []
        self._aspect_ratios = []
        for box_spec in self._box_specs:
            scales, aspect_ratios = zip(*box_spec)
            self._scales.append(scales)
            self._aspect_ratios.append(aspect_ratios)

    def generate(self, feature_map_shape_list, im_height=1, im_width=1):
        """Generates a collection of bounding boxes to be used as anchors.

        The number of anchors generated for a single grid with shape MxM where we
        place k boxes over each grid center is k*M^2 and thus the total number of
        anchors is the sum over all grids. In our box_specs_list example
        (see the constructor docstring), we would place two boxes over each grid
        point on an 8x8 grid and three boxes over each grid point on a 4x4 grid and
        thus end up with 2*8^2 + 3*4^2 = 176 anchors in total. The layout of the
        output anchors follows the order of how the grid sizes and box_specs are
        specified (with box_spec index varying the fastest, followed by width
        index, then height index, then grid index).

        Args:
            feature_map_shape_list: list of pairs of convnet layer resolutions in the
                format [(height_0, width_0), (height_1, width_1), ...]. For example,
                setting feature_map_shape_list=[(8, 8), (7, 7)] asks for anchors that
                correspond to an 8x8 layer followed by a 7x7 layer.
            im_height: the height of the image to generate the grid for. If both
                im_height and im_width are 1, the generated anchors default to
                normalized coordinates, otherwise absolute coordinates are used for the
                grid.
            im_width: the width of the image to generate the grid for. If both
                im_height and im_width are 1, the generated anchors default to
                normalized coordinates, otherwise absolute coordinates are used for the
                grid.

        Returns:
            boxes_list: a list of BoxLists each holding anchor boxes corresponding to
                the input feature map shapes.

        Raises:
            ValueError: if feature_map_shape_list, box_specs_list do not have the same
                length.
            ValueError: if feature_map_shape_list does not consist of pairs of
                integers
        """
        if not (isinstance(feature_map_shape_list, list)
                        and len(feature_map_shape_list) == len(self._box_specs)):
            raise ValueError('feature_map_shape_list must be a list with the same '
                                             'length as self._box_specs')
        if not all([isinstance(list_item, tuple) and len(list_item) == 2
                                for list_item in feature_map_shape_list]):
            raise ValueError('feature_map_shape_list must be a list of pairs.')

        im_height = tf.to_float(im_height)
        im_width = tf.to_float(im_width)

        anchor_strides = [(1.0 / tf.to_float(pair[0]), 1.0 / tf.to_float(pair[1]))
                                            for pair in feature_map_shape_list]
        anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1])
                                            for stride in anchor_strides]

        anchor_grid_list = []
        anchor_indices_list = []
        min_im_shape = tf.minimum(im_height, im_width)
        scale_height = min_im_shape / im_height
        scale_width = min_im_shape / im_width
        base_anchor_size = [
                scale_height * self._base_anchor_size[0],
                scale_width * self._base_anchor_size[1]
        ]
        for feature_map_index, (grid_size, scales, aspect_ratios, stride,
                                offset) in enumerate(
                                    zip(feature_map_shape_list, self._scales,
                                        self._aspect_ratios, anchor_strides,
                                        anchor_offsets)):
            tiled_anchors = self.tile_anchors(
                    grid_height=grid_size[0],
                    grid_width=grid_size[1],
                    scales=scales,
                    aspect_ratios=aspect_ratios,
                    base_anchor_size=base_anchor_size,
                    anchor_stride=stride,
                    anchor_offset=offset)
            #num_anchors_in_layer = tiled_anchors.num_boxes_static()
            num_anchors_in_layer = tiled_anchors.get_shape()[0].value
            if num_anchors_in_layer is None:
                num_anchors_in_layer = tf.shape(tiled_anchors)[0]
            anchor_indices = feature_map_index * tf.ones([num_anchors_in_layer])
            #tiled_anchors.add_field('feature_map_index', anchor_indices)
            anchor_grid_list.append(tiled_anchors)
            anchor_indices_list.append(anchor_indices)

        return anchor_grid_list, anchor_indices

    def num_anchors_per_location(self):
        """Returns the number of anchors per spatial location.

        Returns:
            a list of integers, one for each expected feature map to be passed to
            the Generate function.
        """
        return [len(box_specs) for box_specs in self._box_specs]


class SSDAnchorGenerator(MultipleGridAnchorGenerator):

    def __init__(self,
                num_layers=6,
                min_scale=0.2,
                max_scale=0.95,
                scales=None,
                aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),
                interpolated_scale_aspect_ratio=1.0,
                base_anchor_size=None,
                reduce_boxes_in_lowest_layer=True):
        if base_anchor_size is None:
            base_anchor_size = [1.0, 1.0]
        base_anchor_size = tf.constant(base_anchor_size, dtype=tf.float32)
        box_specs_list = []
        if scales is None or not scales:
            scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
                                for i in range(num_layers)] + [1.0]
        else:
            # Add 1.0 to the end, which will only be used in scale_next below and used
            # for computing an interpolated scale for the largest scale in the list.
            scales += [1.0]

        for layer, scale, scale_next in zip(
                range(num_layers), scales[:-1], scales[1:]):
            layer_box_specs = []
            if layer == 0 and reduce_boxes_in_lowest_layer:
                layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
            else:
                for aspect_ratio in aspect_ratios:
                    layer_box_specs.append((scale, aspect_ratio))
                # Add one more anchor, with a scale between the current scale, and the
                # scale for the next layer, with a specified aspect ratio (1.0 by
                # default).
                if interpolated_scale_aspect_ratio > 0.0:
                    layer_box_specs.append((np.sqrt(scale*scale_next),
                                            interpolated_scale_aspect_ratio))
            box_specs_list.append(layer_box_specs)

        super(SSDAnchorGenerator, self).__init__(box_specs_list, base_anchor_size)

