#coding=utf-8
#date: 2018-05-03
#author: hyxbiao

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.estimator import BaseEstimator
from imgcv.utils import visualization_utils as vis_utils


class Estimator(BaseEstimator):

    def __init__(self, flags, multi_gpu=False):
        super(Estimator, self).__init__(flags)
        self.multi_gpu = multi_gpu

    def optimizer_fn(self, learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        return optimizer

    def new_model(self, features, labels, mode, params):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params):

        model = self.new_model(features, labels, mode, params)

        tf.logging.info('mode: %s', mode)
        prediction_dict, detection_dict = model(features, mode == tf.estimator.ModeKeys.TRAIN)

        predictions = detection_dict

        if mode == tf.estimator.ModeKeys.PREDICT:
            # Return the predictions and the specification for serving a SavedModel
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })

        #visualization
        #image_with_box = tf.image.draw_bounding_boxes(features['images'], detection_dict['detection_boxes'])
        #tf.summary.image('predict_images', image_with_box)
        eval_dict = {
            'images': features['images'],
            'groundtruth_boxes': labels['groundtruth_boxes'],
            'groundtruth_classes': labels['groundtruth_classes'],
            'detection_boxes': detection_dict['detection_boxes'],
            'detection_classes': detection_dict['detection_classes'],
            'detection_scores': detection_dict['detection_scores'],
            'num_detections': detection_dict['num_detections'],
        }
        category_index = {1: {'id': 1, 'name': 'object'}}
        draw_image_list = (
            vis_utils.draw_side_by_side_evaluation_images(
                eval_dict, category_index,
                max_images=3,
                max_boxes_to_draw=3,
                min_score_thresh=0.2))
        for i, draw_image in enumerate(draw_image_list):
            tf.summary.image('Detections_Left_Groundtruth_Right_{}'.format(i),
                             draw_image)

        tensors_to_log = model.tensors_to_log

        #loss
        loss_dict = model.loss(prediction_dict, labels['groundtruth_boxes'], labels['groundtruth_classes'])

        for loss_tensor in loss_dict.values():
            tf.losses.add_loss(loss_tensor)

        loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', loss)
        tensors_to_log[loss.op.name] = loss

        reg_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar('reg_loss', reg_loss)
        tensors_to_log[reg_loss.op.name] = reg_loss

        losses = tf.losses.get_losses()
        for loss_tensor in losses:
            tensors_to_log[loss_tensor.op.name] = loss_tensor
            tf.summary.scalar(loss_tensor.op.name, loss_tensor)

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()

            learning_rate = self.learning_rate_fn(global_step)

            # Create a tensor named learning_rate for logging purposes
            tf.identity(learning_rate, name='learning_rate')
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = self.optimizer_fn(learning_rate)
            # If we are running multi-GPU, we need to wrap the optimizer.
            if self.multi_gpu:
                optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
        else:
            train_op = None

        training_hooks = []
        training_hooks.append(tf.train.LoggingTensorHook(
            tensors=tensors_to_log,
            every_n_iter=1))

        #TODO: add eval
        metrics = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            training_hooks=training_hooks,
            eval_metric_ops=metrics)

