from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import re

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
slim = tf.contrib.slim

from imgcv.models.model import Model as BaseModel
from imgcv.models.ssd import ssd_vgg
from imgcv.utils import ops
from imgcv.utils import shape_utils
from imgcv.detection import losses
from imgcv.detection import matcher
from imgcv.detection import box_predictor


class SSDVGGModel(BaseModel):
    """Base class for building the Vgg Model."""


    def __init__(self,
            num_classes,
            anchor,
            box_coder,
            score_fn='sigmoid',
            data_format=None,
            weight_decay=0.0005):
        super(SSDVGGModel, self).__init__()

        if not data_format:
            data_format = (
                    'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.num_classes = num_classes
        self.anchor = anchor
        self.data_format = data_format
        self._weight_decay = weight_decay

        self._extract_features_scope = 'FeatureExtractor'

        self._unmatched_cls_target = tf.constant([1] + self.num_classes * [0],
                                           tf.float32)
        self._negative_class_weight = 1.0

        self._box_coder = box_coder

        if score_fn == 'sigmoid':
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

        for model_var in slim.get_model_variables():
            tf.summary.histogram('ModelVars/' + model_var.op.name, model_var)

        return prediction_dict, detection_dict

    def preprocess(self, resized_inputs):
        return (2.0 / 255.0) * resized_inputs - 1.0
        #return resized_inputs

    def predict(self, resized_inputs, training):
        #get and preprocess inputs
        preprocessed_inputs = self.preprocess(resized_inputs)

        data_format = 'NHWC' if self.data_format == 'channels_last' else 'NCHW'
        batchnorm_updates_collections = tf.GraphKeys.UPDATE_OPS
        with slim.arg_scope([slim.batch_norm],
                            is_training=training,
                            updates_collections=batchnorm_updates_collections):
            with tf.variable_scope(None, self._extract_features_scope,
                                   [preprocessed_inputs]):
                #exract feature maps
                feature_maps = ssd_vgg.extract_ssd_300_vgg_features(
                        inputs=preprocessed_inputs,
                        is_training=training,
                        data_format=data_format,
                        weight_decay=self._weight_decay)

            #generate anchor
            num_anchors_per_location = self.anchor.num_anchors_per_location()
            feature_masks_list = []
            feature_maps_shape = []
            for i, feature_map, num_per_location in zip(
                    range(len(feature_maps)), feature_maps, num_anchors_per_location):
                shape = shape_utils.combined_static_and_dynamic_shape(
                            feature_map)
                feature_maps_shape.append((shape[1], shape[2]))
                feature_masks_list.append(tf.zeros([shape[1]*shape[2]*num_per_location]) + i)
            feature_masks = tf.concat(feature_masks_list, axis=0)

            image_shape = shape_utils.combined_static_and_dynamic_shape(
                    preprocessed_inputs)
            tf.logging.info('preprocessed image shape: %s', image_shape)

            anchor_grid_list, anchor_indices_list = self.anchor.generate(
                    feature_maps_shape, image_shape[1], image_shape[2])

            anchor_grids = tf.concat(anchor_grid_list, 0)
            anchor_indices = tf.concat(anchor_indices_list, 0)

            #predict
            _box_predictor = box_predictor.ConvolutionalBoxPredictor(
                is_training=training,
                num_classes=self.num_classes,
                num_predictions_per_location_list=num_anchors_per_location,
                kernel_size=3,
                box_code_size=4,
                use_depthwise=True,
            )
            box_encodings_list, class_predictions_list = _box_predictor.predict(
                    feature_maps)
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
                'feature_masks': feature_masks,
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

            indices = tf.range(num_boxes)

            nms_boxes_list = []
            nms_scores_list = []
            nms_classes_list = []
            nms_indices_list = []
            for class_idx in range(num_classes):
                class_scores = tf.reshape(
                    tf.slice(scores, [0, class_idx], tf.stack([num_scores, 1])), [-1])

                high_score_indices = tf.cast(tf.reshape(
                    tf.where(tf.greater(class_scores, score_threshold)),
                    [-1]), tf.int32)
                boxes_filtered = tf.gather(boxes, high_score_indices)
                class_scores_filtered = tf.gather(class_scores, high_score_indices)
                indices_filtered = tf.gather(indices, high_score_indices)

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
                nms_indices = tf.gather(indices_filtered, selected_indices)

                nms_boxes_list.append(nms_boxes)
                nms_scores_list.append(nms_scores)
                nms_classes_list.append(tf.zeros_like(nms_scores) + class_idx)
                nms_indices_list.append(nms_indices)
            selected_boxes = tf.concat(nms_boxes_list, 0)
            selected_scores = tf.concat(nms_scores_list, 0)
            selected_classes = tf.concat(nms_classes_list, 0)
            selected_indices = tf.concat(nms_indices_list, 0)

            if max_total_size > 0:
                top_k = tf.minimum(max_total_size,
                                   tf.shape(selected_boxes)[0])
            else:
                top_k = tf.shape(selected_boxes)[0]
            sorted_scores, sorted_indices = tf.nn.top_k(selected_scores,
                    k=top_k, sorted=True)
            sorted_boxes = tf.gather(selected_boxes, sorted_indices)
            sorted_classes = tf.gather(selected_classes, sorted_indices)
            sorted_index = tf.gather(selected_indices, sorted_indices)
            return sorted_boxes, sorted_scores, sorted_classes, sorted_index, top_k

    def batch_non_max_suppression(self,
            detection_boxes,
            detection_scores,
            parallel_iterations=32,
            score_threshold=1e-8,
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
                    dtype=(tf.float32, tf.float32, tf.float32, tf.int32, tf.int32),
                    parallel_iterations=parallel_iterations)

            batch_sorted_boxes, batch_sorted_scores, \
                    batch_sorted_classes, batch_sorted_indices, \
                    batch_num_detections = batch_outputs
            return (batch_sorted_boxes, batch_sorted_scores,
                    batch_sorted_classes, batch_sorted_indices, batch_num_detections)

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
            feature_maps = prediction_dict['feature_maps']
            feature_masks = prediction_dict['feature_masks']

            #decode
            detection_boxes = self.batch_decode(anchors, box_encodings)
            tf.logging.info('detection_boxes: %s', detection_boxes)

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
                    nmsed_indices, num_detections) = self.batch_non_max_suppression(
                         detection_boxes,
                         detection_scores)
            tf.logging.info('nmsed boxes: %s', nmsed_boxes)
            tf.logging.info('nmsed scores: %s', nmsed_scores)
            tf.logging.info('nmsed classes: %s', nmsed_classes)
            tf.logging.info('nmsed indices: %s', nmsed_indices)
            nmsed_masks = tf.gather(feature_masks, nmsed_indices)
            count_list = []
            for i in range(len(feature_maps)):
                count = tf.reduce_sum(tf.to_int32(tf.equal(nmsed_masks, i)))
                count_list.append(count)
            nmsed_feature_distribute = tf.to_int32(count_list)
            self.tensors_to_log['nmsed_feature'] = nmsed_feature_distribute
            #y, _, count = tf.unique_with_counts(tf.reshape(tf.to_int32(nmsed_classes), [-1]))
            #classes_distribute = tf.stack([y, count])
            count = tf.bincount(
                    tf.reshape(tf.to_int32(nmsed_classes), [-1]))
            self.tensors_to_log['top_classes_count'] = count

            #tensor log
            max_scores = tf.squeeze(tf.slice(nmsed_scores,
                    [0, 0], [-1, 1]), name='max_scores')
            #max_scores = tf.slice(nmsed_scores,
            #        [0, 0], [-1, 5], name='max_scores')
            max_classes = tf.squeeze(tf.slice(nmsed_classes,
                    [0, 0], [-1, 1]), name='max_classes')
            #max_classes = tf.slice(nmsed_classes,
            #        [0, 0], [-1, 5], name='max_classes')
            self.tensors_to_log[max_scores.op.name] = max_scores
            self.tensors_to_log[max_classes.op.name] = max_classes

            detection_dict = {
                'detection_boxes': nmsed_boxes,
                'detection_scores': nmsed_scores,
                'detection_classes': nmsed_classes,
                'num_detections': tf.to_float(num_detections)
            }
            for name in detection_dict:
                tf.logging.info('[detection] %s: %s', name, detection_dict[name])
            return detection_dict

    def assign_targets(self, anchors, gt_boxes, gt_classes, gt_labels, scope=None):

        num_gt_boxes = tf.shape(gt_boxes)[0]
        if gt_classes is None:
            raise ValueError('Not support gt classes is None')
            gt_classes = tf.ones(tf.expand_dims(num_gt_boxes, 0))
            gt_classes = tf.expand_dims(gt_classes, -1)

        unmatched_shape_assert = shape_utils.assert_shape_equal(
            shape_utils.combined_static_and_dynamic_shape(gt_classes)[1:],
            shape_utils.combined_static_and_dynamic_shape(
                self._unmatched_cls_target))
        labels_and_box_shapes_assert = shape_utils.assert_shape_equal(
            shape_utils.combined_static_and_dynamic_shape(
                gt_classes)[:1],
            shape_utils.combined_static_and_dynamic_shape(
                gt_boxes)[:1])

        with tf.name_scope(scope, 'AssignTargets'):
            with tf.control_dependencies(
                [unmatched_shape_assert, labels_and_box_shapes_assert]):
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

                cls_labels = matcher.gather_based_on_match(
                        matches, gt_labels, unmatched_value=0, ignored_value=0)


                #regression weight
                groundtruth_weights = tf.ones([num_gt_boxes], dtype=tf.float32)
                reg_weights = matcher.gather_based_on_match(
                    matches, groundtruth_weights, unmatched_value=0., ignored_value=0.)

                #classes weight
                cls_weights = matcher.gather_based_on_match(
                    matches,
                    groundtruth_weights,
                    unmatched_value=self._negative_class_weight,
                    ignored_value=0.)

                return cls_targets, cls_weights, reg_targets, reg_weights, \
                        cls_labels, matches

    def apply_hard_mining(self, location_losses, cls_losses, prediction_dict, matches_list):
        anchors = prediction_dict['anchors']
        box_encodings = prediction_dict['box_encodings']

        detection_boxes = self.batch_decode(anchors, box_encodings)
        return self._hard_example_miner(
            location_losses=location_losses,
            cls_losses=cls_losses,
            detection_boxes=detection_boxes,
            matches_list=matches_list)

    def loss(self, prediction_dict, labels, scope=None):
        with tf.name_scope(scope, 'Loss', prediction_dict.values()):
            groundtruth_boxes_list = labels['groundtruth_boxes']
            groundtruth_classes_list = labels['groundtruth_classes']
            groundtruth_labels_list = labels['groundtruth_labels']
            anchors = prediction_dict['anchors']
            box_encodings = prediction_dict['box_encodings']
            class_predictions = prediction_dict['class_predictions_with_background']
            feature_maps = prediction_dict['feature_maps']
            feature_masks = prediction_dict['feature_masks']

            groundtruth_classes_with_background_list = [
                tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT')
                for one_hot_encoding in groundtruth_classes_list
            ]

            cls_targets_list = []
            cls_weights_list = []
            reg_targets_list = []
            reg_weights_list = []
            cls_labels_list = []
            matches_list = []
            for gt_boxes, gt_classes, gt_lables in zip(
                    groundtruth_boxes_list, groundtruth_classes_with_background_list,
                    groundtruth_labels_list):
                cls_targets, cls_weights, reg_targets, reg_weights, \
                        cls_labels, matches = self.assign_targets(
                        anchors, gt_boxes, gt_classes, gt_lables)
                cls_targets_list.append(cls_targets)
                cls_weights_list.append(cls_weights)
                reg_targets_list.append(reg_targets)
                reg_weights_list.append(reg_weights)
                cls_labels_list.append(cls_labels)
                matches_list.append(matches)

            batch_cls_targets = tf.stack(cls_targets_list)
            batch_cls_weights = tf.stack(cls_weights_list)
            batch_reg_targets = tf.stack(reg_targets_list)
            batch_reg_weights = tf.stack(reg_weights_list)
            batch_cls_labels = tf.stack(cls_labels_list)

            tf.logging.info('batch_cls_targets: %s', batch_cls_targets)
            tf.logging.info('batch_cls_weights: %s', batch_cls_weights)
            tf.logging.info('batch_reg_targets: %s', batch_reg_targets)
            tf.logging.info('batch_reg_weights: %s', batch_reg_weights)
            tf.logging.info('batch_cls_labels: %s', batch_cls_labels)
            self._summarize_target_assignment(groundtruth_boxes_list, matches_list)

            #loss
            location_losses = self._localization_loss(
                box_encodings,
                batch_reg_targets,
                ignore_nan_targets=True,
                weights=batch_reg_weights)
            cls_losses = ops.reduce_sum_trailing_dimensions(
                self._classification_loss(
                    class_predictions,
                    batch_cls_targets,
                    weights=batch_cls_weights),
                ndims=2)
            tf.logging.info('location_losses: %s', location_losses)
            tf.logging.info('cls_losses: %s', cls_losses)

            if self._hard_example_miner:
                (localization_loss, classification_loss, selected_masks) = self.apply_hard_mining(
                    location_losses, cls_losses, prediction_dict, matches_list)
                self._hard_example_miner.summarize()
                self.tensors_to_log.update(self._hard_example_miner.tensors_to_log())
            else:
                selected_masks = tf.ones_like(location_losses, dtype=bool)
                localization_loss = tf.reduce_sum(location_losses)
                classification_loss = tf.reduce_sum(cls_losses)
            normalizer = tf.maximum(tf.to_float(tf.reduce_sum(batch_reg_weights)),
                                    1.0)

            #tensor to log
            feature_loc_losses = tf.reduce_sum(
                    tf.where(selected_masks,
                        location_losses, tf.zeros_like(location_losses)),
                    axis=0)
            feature_cls_losses = tf.reduce_sum(
                    tf.where(selected_masks,
                        cls_losses, tf.zeros_like(cls_losses)),
                    axis=0)
            match_counts = tf.reduce_sum(batch_reg_weights, axis=0)
            selected_counts = tf.reduce_sum(tf.to_int32(selected_masks), axis=0)
            feature_loc_losses_list = []
            feature_cls_losses_list = []
            counts_list = []
            selected_counts_list = []
            for i, feature_map in enumerate(feature_maps):
                feature_mask = tf.equal(feature_masks, i)

                selected_count = tf.reduce_sum(tf.where(feature_mask,
                    selected_counts, tf.zeros_like(selected_counts)))
                selected_counts_list.append(selected_count)

                feature_loc_losses_list.append(tf.reduce_sum(tf.where(feature_mask,
                    feature_loc_losses, tf.zeros_like(feature_loc_losses))) / tf.to_float(selected_count))
                feature_cls_losses_list.append(tf.reduce_sum(tf.where(feature_mask,
                    feature_cls_losses, tf.zeros_like(feature_cls_losses))) / tf.to_float(selected_count))

                counts = tf.reduce_sum(tf.where(feature_mask,
                    match_counts, tf.zeros_like(match_counts)))
                counts_list.append(counts)
            feature_losses = tf.stack([
                tf.to_float(feature_loc_losses_list),
                tf.to_float(feature_cls_losses_list)
            ], name='feature_losses')
            counts = tf.stack([
                tf.to_int32(counts_list),
                tf.to_int32(selected_counts_list)
            ], name='counts')
            self.tensors_to_log[feature_losses.op.name] = feature_losses
            self.tensors_to_log[counts.op.name] = counts

            class_loc_losses_list = []
            class_cls_losses_list = []
            class_counts_list = []
            for i in range(self.num_classes + 1):
                class_mask = tf.logical_and(selected_masks, tf.equal(batch_cls_labels, i))
                selected_count = tf.reduce_sum(tf.to_int32(class_mask))
                class_counts_list.append(selected_count)

                class_loc_losses_list.append(tf.reduce_sum(tf.where(class_mask,
                    location_losses, tf.zeros_like(location_losses))) / tf.to_float(selected_count))
                class_cls_losses_list.append(tf.reduce_sum(tf.where(class_mask,
                    cls_losses, tf.zeros_like(cls_losses))) / tf.to_float(selected_count))
            label_count_loss = tf.stack([
                tf.to_float(class_counts_list),
                tf.to_float(class_loc_losses_list),
                tf.to_float(class_cls_losses_list),
            ], name='label_count_loss')
            self.tensors_to_log[label_count_loss.op.name] = label_count_loss

            count = tf.bincount(
                    tf.reshape(tf.to_int32(tf.concat(groundtruth_labels_list, 0)), [-1]))
            self.tensors_to_log['gt_classes_count'] = count

            #sigma * 1/N
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

    def _summarize_target_assignment(self, groundtruth_boxes_list, matches_list):
        """Creates tensorflow summaries for the input boxes and anchors.

        This function creates four summaries corresponding to the average
        number (over images in a batch) of (1) groundtruth boxes, (2) anchors
        marked as positive, (3) anchors marked as negative, and (4) anchors marked
        as ignored.

        Args:
            groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]
                containing corners of the groundtruth boxes.
            matches_list: a list of matcher.Match objects encoding the match between
                anchors and groundtruth boxes for each image of the batch,
                with rows of the Match objects corresponding to groundtruth boxes
                and columns corresponding to anchors.
        """
        num_boxes_per_image = tf.stack(
                [tf.shape(x)[0] for x in groundtruth_boxes_list])
        batch_matches = tf.stack(matches_list)
        pos_anchors_per_image = tf.reduce_sum(tf.to_float(tf.greater_equal(batch_matches, 0)), axis=1)
        neg_anchors_per_image = tf.reduce_sum(tf.to_float(tf.equal(batch_matches, -1)), axis=1)
        ignored_anchors_per_image = tf.reduce_sum(tf.to_float(tf.equal(batch_matches, -2)), axis=1)
        tf.summary.scalar('AvgNumGroundtruthBoxesPerImage',
                                            tf.reduce_mean(tf.to_float(num_boxes_per_image)),
                                            family='TargetAssignment')
        tf.summary.scalar('AvgNumPositiveAnchorsPerImage',
                                            tf.reduce_mean(tf.to_float(pos_anchors_per_image)),
                                            family='TargetAssignment')
        tf.summary.scalar('AvgNumNegativeAnchorsPerImage',
                                            tf.reduce_mean(tf.to_float(neg_anchors_per_image)),
                                            family='TargetAssignment')
        tf.summary.scalar('AvgNumIgnoredAnchorsPerImage',
                                            tf.reduce_mean(tf.to_float(ignored_anchors_per_image)),
                                            family='TargetAssignment')

    def restore_map(self,
                    fine_tune_checkpoint_type='detection',
                    load_all_detection_checkpoint_vars=False):
        """Returns a map of variables to load from a foreign checkpoint.

        See parent class for details.

        Args:
            fine_tune_checkpoint_type: whether to restore from a full detection
                checkpoint (with compatible variable names) or to restore from a
                classification checkpoint for initialization prior to training.
                Valid values: `detection`, `classification`. Default 'detection'.
            load_all_detection_checkpoint_vars: whether to load all variables (when
                 `from_detection_checkpoint` is True). If False, only variables within
                 the appropriate scopes are included. Default False.

        Returns:
            A dict mapping variable names (to load from a checkpoint) to variables in
            the model graph.
        Raises:
            ValueError: if fine_tune_checkpoint_type is neither `classification`
                nor `detection`.
        """
        if fine_tune_checkpoint_type not in ['detection', 'classification']:
            raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
                    fine_tune_checkpoint_type))
        variables_to_restore = {}
        for variable in tf.global_variables():
            var_name = variable.op.name
            if (fine_tune_checkpoint_type == 'detection' and
                    load_all_detection_checkpoint_vars):
                variables_to_restore[var_name] = variable
            else:
                if var_name.startswith(self._extract_features_scope):
                    if fine_tune_checkpoint_type == 'classification':
                        var_name = (
                                re.split('^' + self._extract_features_scope + '/',
                                                 var_name)[-1])
                    variables_to_restore[var_name] = variable

        return variables_to_restore
