from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
slim = tf.contrib.slim

from imgcv.models.model import Model as BaseModel
from imgcv.models.vgg import vgg
from imgcv.utils import preprocess as pp
from imgcv.utils import shape_utils


class SSDVGGModel(BaseModel):
    """Base class for building the Vgg Model."""


    def __init__(self,
            num_classes,
            anchor,
            box_coder,
            data_format=None,
            weight_decay=0.0005,
            dropout_keep_prob=0.5):
        super(SSDVGGModel, self).__init__()

        if not data_format:
            data_format = (
                    'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.num_classes = num_classes
        self.anchor = anchor
        self.data_format = data_format
        self.weight_decay = weight_decay
        self.dropout_keep_prob = dropout_keep_prob

        self._matched_threshold = 0.5
        self._unmatched_threshold = 0.5

        self._unmatched_cls_target = tf.constant([1] + self.num_classes * [0],
                                           tf.float32)
        self._negative_class_weight = 1.0

        self._box_coder = box_coder

    def get_inputs(self, inputs):
        raise NotImplementedError

    def preprocess(self, resized_inputs):
        return (2.0 / 255.0) * resized_inputs - 1.0

    @add_arg_scope
    def pad2d(self,
              inputs,
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

    def ssd_arg_scope(self):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        data_format = 'NHWC' if self.data_format == 'channels_last' else 'NCHW'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                data_format=data_format):
                with slim.arg_scope([self.pad2d],
                                    data_format=data_format) as sc:
                    return sc

    def ssd_300_vgg(self,
            inputs,
            is_training=True,
            dropout_keep_prob=0.5,
            scope=None
            ):
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
                    net = self.pad2d(net, pad=(1, 1))
                    net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                with tf.variable_scope('block9'):
                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = self.pad2d(net, pad=(1, 1))
                    net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                with tf.variable_scope('block10'):
                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
                with tf.variable_scope('block11'):
                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points

    def extract_features(self, preprocessed_inputs, training):
        tf.logging.info(preprocessed_inputs)
        #with slim.arg_scope(vgg.vgg_arg_scope()):
        scope = 'ssd_300_vgg'
        arg_scope = self.ssd_arg_scope()
        with slim.arg_scope(arg_scope):
            _, end_points = self.ssd_300_vgg(preprocessed_inputs,
                                            is_training=training,
                                            dropout_keep_prob=self.dropout_keep_prob,
                                            scope=scope)
        image_features = []
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
        image_features += [end_points['%s/%s' % (scope, layer)] for layer in layers]

        for image_feature in image_features:
            tf.logging.info(image_feature)
        return image_features

    def box_predict(self, image_features, num_predictions_per_location_list):
        kernel_size = 3
        box_code_size = 4
        box_encodings_list = []
        class_predictions_list = []
        for (i, image_feature, num_predictions_per_location) in zip(
                    range(len(image_features)), image_features, num_predictions_per_location_list):
            # Add a slot for the background class.
            num_class_slots = self.num_classes + 1
            net = image_feature
            with tf.variable_scope('BoxPredictor_{}'.format(i)):
                with slim.arg_scope([slim.conv2d], activation_fn=None,
                                    normalizer_fn=None, normalizer_params=None,
                                    weights_regularizer=slim.l2_regularizer(self.weight_decay),
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    biases_initializer=tf.zeros_initializer()):
                    box_encodings = slim.conv2d(
                        net, num_predictions_per_location * box_code_size,
                        [kernel_size, kernel_size],
                        scope='BoxEncodingPredictor')
                    class_predictions_with_background = slim.conv2d(
                        net, num_predictions_per_location * num_class_slots,
                        [kernel_size, kernel_size],
                        scope='ClassPredictor',
                        biases_initializer=tf.zeros_initializer())

            shape = image_feature.shape.as_list()
            box_encodings = tf.reshape(
                box_encodings, tf.stack([shape[0],
                                         shape[1] * shape[2] * num_predictions_per_location,
                                         1, box_code_size]))
            box_encodings_list.append(box_encodings)
            class_predictions_with_background = tf.reshape(
                class_predictions_with_background,
                tf.stack([shape[0],
                          shape[1] * shape[2] * num_predictions_per_location,
                          num_class_slots]))
            class_predictions_list.append(class_predictions_with_background)

        return box_encodings_list, class_predictions_list

    def __call__(self, resized_inputs, training):
        """Add operations to classify a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Set to True to add operations required only when
                training the classifier.

        Returns:
            A logits Tensor with shape [<batch_size>, self.num_classes].
        """
        #get and preprocess inputs
        preprocessed_inputs = self.preprocess(resized_inputs)

        #exract feature maps
        feature_maps = self.extract_features(preprocessed_inputs, training)

        #generate anchor
        feature_maps_shape = []
        for feature_map in feature_maps:
            shape = feature_map.shape.as_list()
            feature_maps_shape.append((shape[1], shape[2]))
        image_shape = preprocessed_inputs.shape.as_list()
        tf.logging.info(feature_maps_shape)
        tf.logging.info(image_shape)

        anchor_grid_list, anchor_indices_list = self.anchor.generate(
                feature_maps_shape, image_shape[1], image_shape[2])

        anchor_grids = tf.concat(anchor_grid_list, 0)
        anchor_indices = tf.concat(anchor_indices_list, 0)

        tf.logging.info(anchor_grids)
        tf.logging.info(anchor_indices)

        #predict
        box_encodings_list, class_predictions_list = self.box_predict(
                feature_maps, self.anchor.num_anchors_per_location())
        tf.logging.info(box_encodings_list)
        tf.logging.info(class_predictions_list)

        box_encodings = tf.squeeze(
            tf.concat(box_encodings_list, axis=1), axis=2)
        class_predictions_with_background = tf.concat(
            class_predictions_list, axis=1)
        predictions_dict = {
            'preprocessed_inputs': preprocessed_inputs,
            'box_encodings': box_encodings,
            'class_predictions_with_background':
            class_predictions_with_background,
            'feature_maps': feature_maps,
            'anchors': anchor_grids,
        }
        tf.logging.info(predictions_dict)
        return predictions_dict

    def _set_values_using_indicator(self, x, indicator, val):
        indicator = tf.cast(indicator, x.dtype)
        return tf.add(tf.multiply(x, 1 - indicator), val * indicator)

    def match(self, anchors, groundtruth_boxes, groundtruth_classes):
        similarity_matrix = pp.box.iou(groundtruth_boxes, anchors)

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

    def gather_based_on_match(self, matches, input_tensor, unmatched_value, ignored_value):
        input_tensor = tf.concat([tf.stack([ignored_value, unmatched_value]),
                                  input_tensor], axis=0)
        gather_indices = tf.maximum(matches + 2, 0)
        gathered_tensor = tf.gather(input_tensor, gather_indices)
        return gathered_tensor

    def loss(self, predictions_dict, groundtruth_boxes_list, groundtruth_classes_list):
        groundtruth_classes_with_background_list = [
            tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT')
            for one_hot_encoding in groundtruth_classes_list
        ]
        anchors = predictions_dict['anchors']
        for gt_boxes, gt_classes in zip(
                groundtruth_boxes_list, groundtruth_classes_with_background_list):
            matches = self.match(anchors, gt_boxes, gt_classes)

            matched_gt_boxes = self.gather_based_on_match(
                    matches,
                    gt_boxes,
                    unmatched_value=tf.zeros(4),
                    ignored_value=tf.zeros(4))

            tf.logging.info(matched_gt_boxes)

            matched_reg_targets = self._box_coder.encode(matched_gt_boxes, anchors)
            reg_targets = tf.where(tf.greater_equal(matches, 0),
                                   matched_reg_targets,
                                   tf.zeros_like(matched_reg_targets))

            tf.logging.info(reg_targets)

            cls_targets = self.gather_based_on_match(
                    matches,
                    gt_classes,
                    unmatched_value=self._unmatched_cls_target,
                    ignored_value=self._unmatched_cls_target)

            tf.logging.info(cls_targets)

            num_gt_boxes = tf.shape(gt_boxes)[0]
            groundtruth_weights = tf.ones([num_gt_boxes], dtype=tf.float32)
            reg_weights = self.gather_based_on_match(
                matches, groundtruth_weights, ignored_value=0., unmatched_value=0.)
            tf.logging.info(reg_weights)

            cls_weights = self.gather_based_on_match(
                matches,
                groundtruth_weights,
                ignored_value=0.,
                unmatched_value=self._negative_class_weight)
            tf.logging.info(cls_weights)

            break
