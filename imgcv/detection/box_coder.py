#!/usr/bin/env python
# encoding: utf-8


import os
import sys

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.utils import preprocess as pp

EPSILON = 1e-8

class BoxCoder(object):

    def encode(self, boxes, anchors):
        raise NotImplementedError

    def decode(self, rel_codes, anchors):
        raise NotImplementedError


class SSDBoxCoder(BoxCoder):
    def __init__(self, scale_factors=[10., 10., 5., 5.]):
        if scale_factors:
            assert len(scale_factors) == 4
            for scalar in scale_factors:
                assert scalar > 0
        self._scale_factors = scale_factors

    def encode(self, boxes, anchors):
        """Encode a box collection with respect to anchor collection.

        Args:
            boxes: BoxList holding N boxes to be encoded.
            anchors: BoxList of anchors.

        Returns:
            a tensor representing N anchor-encoded boxes of the format
            [ty, tx, th, tw].
        """
        # Convert anchors to the center coordinate representation.
        ycenter_a, xcenter_a, ha, wa = pp.box.get_center_coordinates_and_sizes(anchors)
        ycenter, xcenter, h, w = pp.box.get_center_coordinates_and_sizes(boxes)
        # Avoid NaN in division and log below.
        ha += EPSILON
        wa += EPSILON
        h += EPSILON
        w += EPSILON

        tx = (xcenter - xcenter_a) / wa
        ty = (ycenter - ycenter_a) / ha
        tw = tf.log(w / wa)
        th = tf.log(h / ha)
        # Scales location targets as used in paper for joint training.
        if self._scale_factors:
            ty *= self._scale_factors[0]
            tx *= self._scale_factors[1]
            th *= self._scale_factors[2]
            tw *= self._scale_factors[3]
        return tf.transpose(tf.stack([ty, tx, th, tw]))

    def decode(self, rel_codes, anchors):
        """Decode relative codes to boxes.

        Args:
            rel_codes: a tensor representing N anchor-encoded boxes.
            anchors: BoxList of anchors.

        Returns:
            boxes: BoxList holding N bounding boxes.
        """
        ycenter_a, xcenter_a, ha, wa = pp.box.get_center_coordinates_and_sizes(anchors)

        ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
        if self._scale_factors:
            ty /= self._scale_factors[0]
            tx /= self._scale_factors[1]
            th /= self._scale_factors[2]
            tw /= self._scale_factors[3]
        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.
        return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))
