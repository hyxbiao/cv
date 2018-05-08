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

        self._extract_features_scope = 'FeatureExtractor'

        self._unmatched_cls_target = tf.constant([1] + self.num_classes * [0],
                                           tf.float32)
        self._negative_class_weight = 1.0

        self._box_coder = box_coder

        sigmoid = True
        if sigmoid:
            self._classification_loss = losses.WeightedSigmoidClassificationLoss()
            self._tf_score_converter_fn = tf.sigmoid
        else:
            self._classification_loss = losses.WeightedSoftmaxClassificationLoss()
            self._tf_score_converter_fn = tf.nn.softmax
        self._localization_loss = losses.WeightedSmoothL1LocalizationLoss()
        self._localization_loss_weight = 1.0
        self._classification_loss_weight = 1.0
        self._hard_example_miner = losses.HardExampleMiner(
            num_hard_examples=3000,
            iou_threshold=0.99,
            loss_type='cls',
            cls_loss_weight=self._classification_loss_weight,
            loc_loss_weight=self._localization_loss_weight,
            max_negatives_per_positive=3,
            min_negatives_per_image=3)


        self.tensors_to_log = {}

    def __call__(self, inputs, training):
        images = inputs['images']
        prediction_dict = self.predict(images, training)

        detection_dict = self.postprocess(prediction_dict)

        return prediction_dict, detection_dict

    def preprocess(self, resized_inputs):
        return (2.0 / 255.0) * resized_inputs - 1.0

    def box_predict(self, image_features, num_predictions_per_location_list):
        kernel_size = 3
        box_code_size = 4
        box_encodings_list = []
        class_predictions_list = []
        ratio_list = []
        for (i, image_feature, num_predictions_per_location) in zip(
                    range(len(image_features)), image_features, num_predictions_per_location_list):
            # Add a slot for the background class.
            num_class_slots = self.num_classes + 1
            net = image_feature

            ratio = tf.reduce_sum(tf.to_int32(tf.equal(net, 0))) / tf.size(net)
            tf.summary.scalar('FeatureZeroRatio{}'.format(i), ratio)
            ratio_list.append(ratio)

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

        self.tensors_to_log['feature_zero_ratio'] = tf.to_float(ratio_list)

        return box_encodings_list, class_predictions_list

    def predict(self, resized_inputs, training):
        #get and preprocess inputs
        preprocessed_inputs = self.preprocess(resized_inputs)

        data_format = 'NHWC' if self.data_format == 'channels_last' else 'NCHW'
        with tf.variable_scope(None, self._extract_features_scope,
                               [preprocessed_inputs]):
            #exract feature maps
            feature_maps = ssd_vgg.extract_ssd_300_vgg_features(
                    preprocessed_inputs,
                    training,
                    data_format)

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

        debug_tensor = tf.slice(box_encodings, [0, 0, 0], [-1, 1, 1])
        #self.tensors_to_log['box_encodings'] = debug_tensor
        debug_tensor = tf.slice(class_predictions_with_background , [0, 0, 0], [-1, 1, 1])
        #self.tensors_to_log['class_predictions_with_background'] = debug_tensor

        prediction_dict = {
            'preprocessed_inputs': preprocessed_inputs,
            'box_encodings': box_encodings,
            'class_predictions_with_background':
                class_predictions_with_background,
            'feature_maps': feature_maps,
            'anchors': anchor_grids,
        }
        for name in prediction_dict:
            tf.logging.info('[prediction] %s: %s', name, prediction_dict[name])
        return prediction_dict

    def multiclass_non_max_suppression(self,
            boxes, scores,
            score_threshold,
            max_size_per_class,
            max_total_size,
            iou_threshold,
            scope=None):
        with tf.name_scope(scope, 'MultiClassNonMaxSuppression', [boxes, scores]):
            tf.logging.info(boxes)
            tf.logging.info(scores)

            num_boxes = tf.shape(boxes)[0]
            num_scores = tf.shape(scores)[0]
            num_classes = scores.get_shape()[1]

            nms_boxes_list = []
            nms_scores_list = []
            nms_classes_list = []
            for class_idx in range(num_classes):
                class_scores = tf.reshape(
                    tf.slice(scores, [0, class_idx], tf.stack([num_scores, 1])), [-1])

                high_score_indices = tf.cast(tf.reshape(
                    tf.where(tf.greater(class_scores, score_threshold)),
                    [-1]), tf.int32)
                boxes_filtered = tf.gather(boxes, high_score_indices)
                class_scores_filtered = tf.gather(class_scores, high_score_indices)

                #clip window?
                num_boxes_filtered = tf.shape(boxes_filtered)[0]
                max_selection_size = tf.minimum(max_size_per_class,
                                                num_boxes_filtered)
                # Apply NMS algorithm.
                selected_indices = tf.image.non_max_suppression(
                    boxes_filtered,
                    class_scores_filtered,
                    max_selection_size,
                    iou_threshold=iou_threshold)
                nms_boxes = tf.gather(boxes_filtered, selected_indices)
                nms_scores = tf.gather(class_scores_filtered, selected_indices)

                nms_boxes_list.append(nms_boxes)
                nms_scores_list.append(nms_scores)
                nms_classes_list.append(tf.zeros_like(nms_scores) + class_idx)
            selected_boxes = tf.concat(nms_boxes_list, 0)
            selected_scores = tf.concat(nms_scores_list, 0)
            selected_classes = tf.concat(nms_classes_list, 0)

            if max_total_size > 0:
                top_k = tf.minimum(max_total_size,
                                   tf.shape(selected_boxes)[0])
            else:
                top_k = tf.shape(selected_boxes)[0]
            sorted_scores, sorted_indices = tf.nn.top_k(selected_scores,
                    k=top_k, sorted=True)
            sorted_boxes = tf.gather(selected_boxes, sorted_indices)
            sorted_classes = tf.gather(selected_classes, sorted_indices)
            return sorted_boxes, sorted_scores, sorted_classes, top_k

    def batch_non_max_suppression(self,
            detection_boxes,
            detection_scores,
            parallel_iterations=32,
            score_threshold=0.001,
            max_size_per_class=100,
            max_total_size=100,
            iou_threshold=0.6,
            scope=None):
        with tf.name_scope(scope, 'BatchMultiClassNonMaxSuppression'):
            batch_outputs = tf.map_fn(
                    lambda x: self.multiclass_non_max_suppression(x[0], x[1],
                                    score_threshold=score_threshold,
                                    max_size_per_class=max_size_per_class,
                                    max_total_size=max_total_size,
                                    iou_threshold=iou_threshold),
                    elems=(detection_boxes, detection_scores),
                    dtype=(tf.float32, tf.float32, tf.float32, tf.int32),
                    parallel_iterations=parallel_iterations)

            batch_sorted_boxes, batch_sorted_scores, \
                    batch_sorted_classes, batch_num_detections = batch_outputs
            return (batch_sorted_boxes, batch_sorted_scores,
                    batch_sorted_classes, batch_num_detections)

    def batch_decode(self, anchors, box_encodings):
        combined_shape = shape_utils.combined_static_and_dynamic_shape(
                box_encodings)
        batch_size = combined_shape[0]
        tiled_anchors = tf.tile(
                tf.expand_dims(anchors, 0), [batch_size, 1, 1])
        tiled_anchors = tf.reshape(tiled_anchors, [-1, 4])

        detection_boxes = self._box_coder.decode(
                tf.reshape(box_encodings, [-1, 4]),
                tiled_anchors)
        detection_boxes = tf.reshape(detection_boxes, tf.stack(
                [combined_shape[0], combined_shape[1], 4]))

        return detection_boxes

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
            detection_boxes = self.batch_decode(anchors, box_encodings)
            tf.logging.info('detection_boxes: %s', detection_boxes)
            debug_detection_boxes = tf.slice(detection_boxes, [0, 0, 0],
                                                                        [-1, 1, 1])
            #self.tensors_to_log[debug_detection_boxes.op.name] = debug_detection_boxes

            #score
            detection_scores_with_background = self._tf_score_converter_fn(
                    class_predictions, name='convert_scores')
            detection_scores = tf.slice(detection_scores_with_background, [0, 0, 1],
                                                                        [-1, -1, -1])
            tf.logging.info('detection_scores: %s', detection_scores)
            debug_detection_scores = tf.slice(detection_scores, [0, 0, 0],
                                                                        [-1, 1, 1])
            #self.tensors_to_log[debug_detection_scores.op.name] = debug_detection_scores

            #nms
            (nmsed_boxes, nmsed_scores, nmsed_classes,
                    num_detections) = self.batch_non_max_suppression(
                         detection_boxes,
                         detection_scores)

            #tensor log
            max_scores = tf.squeeze(tf.slice(nmsed_scores,
                    [0, 0], [-1, 1]), name='max_scores')
            self.tensors_to_log[max_scores.op.name] = max_scores

            detection_dict = {
                'detection_boxes': nmsed_boxes,
                'detection_scores': nmsed_scores,
                'detection_classes': nmsed_classes,
                'num_detections': tf.to_float(num_detections)
            }
            for name in detection_dict:
                tf.logging.info('[detection] %s: %s', name, detection_dict[name])
            return detection_dict

    def assign_targets(self, anchors, gt_boxes, gt_classes):
        #match anchors and gt_boxes
        argmax_matcher = matcher.ArgMaxMatcher()
        matches = argmax_matcher.match(anchors, gt_boxes, gt_classes)

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
            matches, groundtruth_weights, unmatched_value=0., ignored_value=0.)

        #classes weight
        cls_weights = matcher.gather_based_on_match(
            matches,
            groundtruth_weights,
            unmatched_value=self._negative_class_weight,
            ignored_value=0.)

        return cls_targets, cls_weights, reg_targets, reg_weights, matches

    def apply_hard_mining(self, location_losses, cls_losses, prediction_dict, matches_list):
        anchors = prediction_dict['anchors']
        box_encodings = prediction_dict['box_encodings']
        class_predictions = prediction_dict['class_predictions_with_background']

        class_predictions = tf.slice(
            class_predictions, [0, 0, 1], [-1, -1, -1])

        detection_boxes = self.batch_decode(anchors, box_encodings)
        return self._hard_example_miner(
            location_losses=location_losses,
            cls_losses=cls_losses,
            detection_boxes=detection_boxes,
            matches_list=matches_list)

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
            batch_matches = tf.stack(matches_list)

            tf.logging.info('batch_cls_targets: %s', batch_cls_targets)
            tf.logging.info('batch_cls_weights: %s', batch_cls_weights)
            tf.logging.info('batch_reg_targets: %s', batch_reg_targets)
            tf.logging.info('batch_reg_weights: %s', batch_reg_weights)

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
            tf.logging.info('location_losses: %s', location_losses)
            tf.logging.info('cls_losses: %s', cls_losses)

            if self._hard_example_miner:
                (localization_loss, classification_loss) = self.apply_hard_mining(
                    location_losses, cls_losses, prediction_dict, matches_list)
                self._hard_example_miner.summarize()
                self.tensors_to_log.update(self._hard_example_miner.tensors_to_log())
            else:
                localization_loss = tf.reduce_sum(location_losses)
                classification_loss = tf.reduce_sum(cls_losses)
            normalizer = tf.maximum(tf.to_float(tf.reduce_sum(batch_reg_weights)),
                                    1.0)

            tf.summary.scalar('normalizer', normalizer)

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

