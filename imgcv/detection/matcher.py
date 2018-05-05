#!/usr/bin/env python
# encoding: utf-8


import os
import sys

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.utils import preprocess as pp
from imgcv.utils import shape_utils

class Matcher(object):

    def match(self, anchors, groundtruth_boxes, groundtruth_classes):
        raise NotImplementedError


class ArgMaxMatcher(Matcher):
    def __init__(self):
        self._matched_threshold = 0.5
        self._unmatched_threshold = 0.5

    def match(self, anchors, groundtruth_boxes, groundtruth_classes):
        similarity_matrix = pp.box.iou(groundtruth_boxes, anchors)

        def _match_when_rows_are_empty():
            similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
                similarity_matrix)
            return -1 * tf.ones([similarity_matrix_shape[1]], dtype=tf.int32)

        def _match_when_rows_are_non_empty():
            # Matches for each column
            matches = tf.argmax(similarity_matrix, 0, output_type=tf.int32)

            # Deal with matched and unmatched threshold
            if self._matched_threshold is not None:
                # Get logical indices of ignored and unmatched columns as tf.int64
                matched_vals = tf.reduce_max(similarity_matrix, 0)
                below_unmatched_threshold = tf.greater(self._unmatched_threshold,
                                                     matched_vals)
                between_thresholds = tf.logical_and(
                        tf.greater_equal(matched_vals, self._unmatched_threshold),
                        tf.greater(self._matched_threshold, matched_vals))

                matches = self._set_values_using_indicator(matches,
                                                         below_unmatched_threshold,
                                                         -1)
                matches = self._set_values_using_indicator(matches,
                                                         between_thresholds,
                                                         -2)

            similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
                similarity_matrix)
            force_match_column_ids = tf.argmax(similarity_matrix, 1,
                                               output_type=tf.int32)
            force_match_column_indicators = tf.one_hot(
                force_match_column_ids, depth=similarity_matrix_shape[1])
            force_match_row_ids = tf.argmax(force_match_column_indicators, 0,
                                            output_type=tf.int32)
            force_match_column_mask = tf.cast(
                tf.reduce_max(force_match_column_indicators, 0), tf.bool)
            final_matches = tf.where(force_match_column_mask,
                                     force_match_row_ids, matches)
            return final_matches

        if similarity_matrix.shape.is_fully_defined():
            if similarity_matrix.shape[0].value == 0:
                return _match_when_rows_are_empty()
            else:
                return _match_when_rows_are_non_empty()
        else:
            return tf.cond(
            tf.greater(tf.shape(similarity_matrix)[0], 0),
            _match_when_rows_are_non_empty, _match_when_rows_are_empty)

    def _set_values_using_indicator(self, x, indicator, val):
        indicator = tf.cast(indicator, x.dtype)
        return tf.add(tf.multiply(x, 1 - indicator), val * indicator)


def gather_based_on_match(matches, input_tensor, unmatched_value, ignored_value):
    input_tensor = tf.concat([tf.stack([ignored_value, unmatched_value]),
                              input_tensor], axis=0)
    gather_indices = tf.maximum(matches + 2, 0)
    gathered_tensor = tf.gather(input_tensor, gather_indices)
    return gathered_tensor

