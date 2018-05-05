from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
slim = tf.contrib.slim

from imgcv.models.model import Model as BaseModel
from imgcv.models.ssd import ssd_vgg
from imgcv.utils import ops
from imgcv.utils import shape_utils
from imgcv.detection import losses
from imgcv.detection import matcher


class SSDVGGModel(BaseModel):
    """Base class for building the Vgg Model."""


    def __init__(self,
            num_classes,
            anchor,
            box_coder,
            data_format=None,
            weight_decay=0.0005):
        super(SSDVGGModel, self).__init__()

        if not data_format:
            data_format = (
                    'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.num_classes = num_classes
        self.anchor = anchor
        self.data_format = data_format
        self.weight_decay = weight_decay

        self._unmatched_cls_target = tf.constant([1] + self.num_classes * [0],
                                           tf.float32)
        self._negative_class_weight = 1.0

        self._box_coder = box_coder

        self._classification_loss = losses.WeightedSigmoidClassificationLoss()
        self._localization_loss = losses.WeightedSmoothL1LocalizationLoss()
        self._localization_loss_weight = 1.0
        self._classification_loss_weight = 1.0

        self._tf_score_converter_fn = tf.sigmoid

    def __call__(self, inputs, training):
        prediction_dict = self.predict(inputs, training)

        self.postprocess(prediction_dict)

        return prediction_dict

    def get_inputs(self, inputs):
        raise NotImplementedError

    def preprocess(self, resized_inputs):
        return (2.0 / 255.0) * resized_inputs - 1.0

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

            shape = shape_utils.combined_static_and_dynamic_shape(
                        image_feature)
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

    def predict(self, resized_inputs, training):
        #get and preprocess inputs
        preprocessed_inputs = self.preprocess(resized_inputs)

        #exract feature maps
        data_format = 'NHWC' if self.data_format == 'channels_last' else 'NCHW'
        feature_maps = ssd_vgg.extract_ssd_300_vgg_features(
                preprocessed_inputs,
                training,
                data_format,
                self.weight_decay)

        #generate anchor
        feature_maps_shape = []
        for feature_map in feature_maps:
            shape = shape_utils.combined_static_and_dynamic_shape(
                        feature_map)
            feature_maps_shape.append((shape[1], shape[2]))
        image_shape = shape_utils.combined_static_and_dynamic_shape(
                preprocessed_inputs)
        tf.logging.info('preprocessed image shape: %s', image_shape)

        anchor_grid_list, anchor_indices_list = self.anchor.generate(
                feature_maps_shape, image_shape[1], image_shape[2])

        anchor_grids = tf.concat(anchor_grid_list, 0)
        anchor_indices = tf.concat(anchor_indices_list, 0)

        #predict
        box_encodings_list, class_predictions_list = self.box_predict(
                feature_maps, self.anchor.num_anchors_per_location())
        box_encodings = tf.squeeze(
            tf.concat(box_encodings_list, axis=1), axis=2)
        class_predictions_with_background = tf.concat(
            class_predictions_list, axis=1)

        prediction_dict = {
            'preprocessed_inputs': preprocessed_inputs,
            'box_encodings': box_encodings,
            'class_predictions_with_background':
                class_predictions_with_background,
            'feature_maps': feature_maps,
            'anchors': anchor_grids,
        }
        for name in prediction_dict:
            tf.logging.info('%s: %s', name, prediction_dict[name])
        return prediction_dict

    def postprocess(self, prediction_dict):
        if ('box_encodings' not in prediction_dict or
                'class_predictions_with_background' not in prediction_dict):
            raise ValueError('prediction_dict does not contain expected entries.')
        with tf.name_scope('Postprocessor'):
            preprocessed_images = prediction_dict['preprocessed_inputs']
            anchors = prediction_dict['anchors']
            box_encodings = prediction_dict['box_encodings']
            class_predictions = prediction_dict['class_predictions_with_background']

            #decode
            combined_shape = shape_utils.combined_static_and_dynamic_shape(
                    box_encodings)
            batch_size = combined_shape[0]
            tiled_anchors = tf.tile(
                    tf.expand_dims(anchors, 0), [batch_size, 1, 1])
            tf.logging.info('tiled anchors: %s', tiled_anchors )
            tiled_anchors = tf.reshape(tiled_anchors, [-1, 4])
            tf.logging.info('tiled anchors: %s', tiled_anchors )

            detection_boxes = self._box_coder.decode(
                    tf.reshape(box_encodings, [-1, 4]),
                    tiled_anchors)

            detection_boxes = tf.expand_dims(detection_boxes, axis=2)

            tf.logging.info('detection_boxes: %s', detection_boxes)

            #score
            detection_scores_with_background = self._tf_score_converter_fn(
                    class_predictions, name='convert_scores')
            detection_scores = tf.slice(detection_scores_with_background, [0, 0, 1],
                                                                        [-1, -1, -1])
            tf.logging.info('detection_scores: %s', detection_scores)

            #nms: tf.image.non_max_suppression??
            image_shape = shape_utils.combined_static_and_dynamic_shape(
                    preprocessed_images)

            (nmsed_boxes, nmsed_scores, nmsed_classes, _, nmsed_additional_fields,
                 num_detections) = self._non_max_suppression_fn(
                         detection_boxes,
                         detection_scores,
                         clip_window=self._compute_clip_window(
                                 preprocessed_images, image_shape),
                         additional_fields=None)
            detection_dict = {
                    fields.DetectionResultFields.detection_boxes: nmsed_boxes,
                    fields.DetectionResultFields.detection_scores: nmsed_scores,
                    fields.DetectionResultFields.detection_classes: nmsed_classes,
                    fields.DetectionResultFields.num_detections:
                            tf.to_float(num_detections)
            }
            if (nmsed_additional_fields is not None and
                    fields.BoxListFields.keypoints in nmsed_additional_fields):
                detection_dict[fields.DetectionResultFields.detection_keypoints] = (
                        nmsed_additional_fields[fields.BoxListFields.keypoints])
            return detection_dict

    def assign_targets(self, anchors, gt_boxes, gt_classes):
        #match anchors and gt_boxes
        matcher = matcher.ArgMaxMatcher()
        matches = matcher.match(anchors, gt_boxes, gt_classes)

        #regression target
        matched_gt_boxes = matcher.gather_based_on_match(
                matches,
                gt_boxes,
                unmatched_value=tf.zeros(4),
                ignored_value=tf.zeros(4))

        matched_reg_targets = self._box_coder.encode(matched_gt_boxes, anchors)
        reg_targets = tf.where(tf.greater_equal(matches, 0),
                               matched_reg_targets,
                               tf.zeros_like(matched_reg_targets))

        #classes target
        cls_targets = matcher.gather_based_on_match(
                matches,
                gt_classes,
                unmatched_value=self._unmatched_cls_target,
                ignored_value=self._unmatched_cls_target)


        #regression weight
        num_gt_boxes = tf.shape(gt_boxes)[0]
        groundtruth_weights = tf.ones([num_gt_boxes], dtype=tf.float32)
        reg_weights = matcher.gather_based_on_match(
            matches, groundtruth_weights, ignored_value=0., unmatched_value=0.)

        #classes weight
        cls_weights = matcher.gather_based_on_match(
            matches,
            groundtruth_weights,
            ignored_value=0.,
            unmatched_value=self._negative_class_weight)

        return cls_targets, cls_weights, reg_targets, reg_weights, matches

    def loss(self, prediction_dict, groundtruth_boxes_list, groundtruth_classes_list, scope=None):
        with tf.name_scope(scope, 'Loss', prediction_dict.values()):
            groundtruth_classes_with_background_list = [
                tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT')
                for one_hot_encoding in groundtruth_classes_list
            ]
            anchors = prediction_dict['anchors']

            cls_targets_list = []
            cls_weights_list = []
            reg_targets_list = []
            reg_weights_list = []
            matches_list = []
            for gt_boxes, gt_classes in zip(
                    groundtruth_boxes_list, groundtruth_classes_with_background_list):
                cls_targets, cls_weights, reg_targets, reg_weights, matches = self.assign_targets(
                        anchors, gt_boxes, gt_classes)
                cls_targets_list.append(cls_targets)
                cls_weights_list.append(cls_weights)
                reg_targets_list.append(reg_targets)
                reg_weights_list.append(reg_weights)
                matches_list.append(matches)

            batch_cls_targets = tf.stack(cls_targets_list)
            batch_cls_weights = tf.stack(cls_weights_list)
            batch_reg_targets = tf.stack(reg_targets_list)
            batch_reg_weights = tf.stack(reg_weights_list)
            tf.logging.info(batch_cls_targets)
            tf.logging.info(batch_cls_weights)
            tf.logging.info(batch_reg_targets)
            tf.logging.info(batch_reg_weights)

            location_losses = self._localization_loss(
                prediction_dict['box_encodings'],
                batch_reg_targets,
                ignore_nan_targets=True,
                weights=batch_reg_weights)
            cls_losses = ops.reduce_sum_trailing_dimensions(
                self._classification_loss(
                    prediction_dict['class_predictions_with_background'],
                    batch_cls_targets,
                    weights=batch_cls_weights),
                ndims=2)
            tf.logging.info(location_losses)
            tf.logging.info(cls_losses)

            #TODO: add hard example miner
            localization_loss = tf.reduce_sum(location_losses)
            classification_loss = tf.reduce_sum(cls_losses)
            normalizer = tf.maximum(tf.to_float(tf.reduce_sum(batch_reg_weights)),
                                    1.0)
            localization_loss_normalizer = normalizer
            localization_loss = tf.multiply((self._localization_loss_weight /
                localization_loss_normalizer),
                localization_loss,
                name='localization_loss')
            classification_loss = tf.multiply((self._classification_loss_weight /
                normalizer), classification_loss,
                name='classification_loss')

            loss_dict = {
                str(localization_loss.op.name): localization_loss,
                str(classification_loss.op.name): classification_loss
            }
        return loss_dict

