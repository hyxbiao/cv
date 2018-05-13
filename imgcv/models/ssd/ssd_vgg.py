#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
slim = tf.contrib.slim

def ssd_vgg_arg_scope(data_format='NHWC', weight_decay=0.0005):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([pad2d],
                                data_format=data_format) as sc:
                return sc


@add_arg_scope
def pad2d(inputs,
          pad=(0, 0),
          mode='CONSTANT',
          data_format='NHWC',
          trainable=True,
          scope=None):
    """2D Padding layer, adding a symmetric padding to H and W dimensions.

    Aims to mimic padding in Caffe and MXNet, helping the port of models to
    TensorFlow. Tries to follow the naming convention of `tf.contrib.layers`.

    Args:
      inputs: 4D input Tensor;
      pad: 2-Tuple with padding values for H and W dimensions;
      mode: Padding mode. C.f. `tf.pad`
      data_format:  NHWC or NCHW data format.
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        # Padding shape.
        if data_format == 'NHWC':
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net


def ssd_300_vgg(inputs,
                is_training=True,
                dropout_keep_prob=0.5,
                scope=None):
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d],
                    outputs_collections=end_points_collection):
            # Original VGG-16 blocks.
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # Block 2.
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # Block 3.
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # Block 4.
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # Block 5.
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

            # Additional SSD blocks.
            # Block 6: let's dilate the hell out of it!
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='extra_dropout6')
            # Block 7: 1x1 conv. Because the fuck.
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='extra_dropout7')

            # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
            with tf.variable_scope('block8'):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                net = pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            with tf.variable_scope('block9'):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            with tf.variable_scope('block10'):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            with tf.variable_scope('block11'):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            prefix_len = len(sc.original_name_scope)
            end_points = dict((k[prefix_len:], end_points[k]) for k in end_points)

    return net, end_points

def extract_ssd_300_vgg_features(
        inputs,
        is_training=True,
        data_format='NHWC',
        weight_decay=0.0005,
        dropout_keep_prob=0.5):
    arg_scope = ssd_vgg_arg_scope(data_format, weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        _, end_points = ssd_300_vgg(inputs,
                                    is_training=is_training,
                                    dropout_keep_prob=dropout_keep_prob)
    layers = [
        #'conv1/conv1_2',
        #'conv2/conv2_2',
        #'conv3/conv3_3',
        'conv4/conv4_3',
        #'conv5/conv5_3',
        #'conv6',
        'conv7',
        'block8/conv3x3',
        'block9/conv3x3',
        'block10/conv3x3',
        'block11/conv3x3',
    ]
    image_features = []
    for layer in layers:
        feature_map = end_points[layer]
        tf.summary.histogram('activations/' + layer, feature_map)
        tf.summary.scalar('sparsity/' + layer,
                                        tf.nn.zero_fraction(feature_map))
        image_features.append(feature_map)


    #add l2 normalization in first feature map
    axis = 3 if data_format == 'NHWC' else 1
    #feature_map = image_features[0]
    #image_features[0] = tf.nn.l2_normalize(feature_map, axis=axis)

    return image_features
