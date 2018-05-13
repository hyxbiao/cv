#!/usr/bin/env python
# encoding: utf-8

import os
import sys

import tensorflow as tf    # pylint: disable=g-bad-import-order
slim = tf.contrib.slim

from imgcv.utils import shape_utils

class BoxPredictor(object):
    def __init__(self, is_training, num_classes):
        self._is_training = is_training
        self._num_classes = num_classes

    @property
    def num_classes(self):
        return self._num_classes

    def predict(self, image_features, scope=None, **params):
        if scope is not None:
            with tf.variable_scope(scope):
                return self._predict(image_features, **params)
        return self._predict(image_features, **params)

    def _predict(self, image_features, **params):
        raise NotImplementedError


class ConvolutionalBoxPredictor(BoxPredictor):
    def __init__(self,
                 is_training,
                 num_classes,
                 num_predictions_per_location_list,
                 kernel_size,
                 box_code_size,
                 use_depthwise,
                 ):
        super(ConvolutionalBoxPredictor, self).__init__(is_training, num_classes)
        self._num_predictions_per_location_list = num_predictions_per_location_list
        self._kernel_size = kernel_size
        self._box_code_size = box_code_size
        self._use_depthwise = use_depthwise
        self._weight_decay = 0.00004

    def arg_scope(self, is_training, bn_decay=0.9997, bn_epsilon=0.001):
        batch_norm_params = {
            'center': True,
            'scale': True,
            'decay': bn_decay,
            'epsilon': bn_epsilon,
            'is_training': is_training,
        }
        affected_ops = [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose]
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope(
                affected_ops,
                weights_regularizer=slim.l2_regularizer(self._weight_decay),
                weights_initializer=tf.truncated_normal_initializer(
                    mean=0.0,
                    stddev=0.03),
                activation_fn=tf.nn.relu6,
                normalizer_fn=slim.batch_norm) as sc:
                return sc

    def _predict(self, image_features, **params):
        box_encodings_list = []
        class_predictions_list = []
        arg_scope = self.arg_scope(self._is_training)
        for (i, image_feature, num_predictions_per_location) in zip(
                    range(len(image_features)), image_features, self._num_predictions_per_location_list):
            # Add a slot for the background class.
            num_class_slots = self._num_classes + 1
            net = image_feature

            with tf.variable_scope('BoxPredictor_{}'.format(i)):
                with slim.arg_scope(arg_scope):
                    with slim.arg_scope([slim.conv2d], activation_fn=None,
                                        normalizer_fn=None, normalizer_params=None):
                        if self._use_depthwise:
                            box_encodings = slim.separable_conv2d(
                                net, None, [self._kernel_size, self._kernel_size],
                                padding='SAME', depth_multiplier=1, stride=1,
                                rate=1, scope='BoxEncodingPredictor_depthwise')
                            box_encodings = slim.conv2d(
                                box_encodings,
                                num_predictions_per_location * self._box_code_size, [1, 1],
                                scope='BoxEncodingPredictor')
                            class_predictions_with_background = slim.separable_conv2d(
                                net, None, [self._kernel_size, self._kernel_size],
                                padding='SAME', depth_multiplier=1, stride=1,
                                rate=1, scope='ClassPredictor_depthwise')
                            class_predictions_with_background = slim.conv2d(
                                class_predictions_with_background,
                                num_predictions_per_location * num_class_slots,
                                [1, 1], scope='ClassPredictor')
                        else:
                            box_encodings = slim.conv2d(
                                net, num_predictions_per_location * self._box_code_size,
                                [self._kernel_size, self._kernel_size],
                                scope='BoxEncodingPredictor')
                            class_predictions_with_background = slim.conv2d(
                                net, num_predictions_per_location * num_class_slots,
                                [self._kernel_size, self._kernel_size],
                                scope='ClassPredictor',
                                biases_initializer=tf.zeros_initializer())

            shape = shape_utils.combined_static_and_dynamic_shape(
                        image_feature)
            box_encodings = tf.reshape(
                box_encodings, tf.stack([shape[0],
                                         shape[1] * shape[2] * num_predictions_per_location,
                                         1, self._box_code_size]))
            box_encodings_list.append(box_encodings)
            class_predictions_with_background = tf.reshape(
                class_predictions_with_background,
                tf.stack([shape[0],
                          shape[1] * shape[2] * num_predictions_per_location,
                          num_class_slots]))
            class_predictions_list.append(class_predictions_with_background)

        return box_encodings_list, class_predictions_list


