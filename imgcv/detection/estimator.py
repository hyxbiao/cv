#coding=utf-8
#date: 2018-05-03
#author: hyxbiao

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.estimator import BaseEstimator
from imgcv.utils import visualization_utils as vis_utils


class Estimator(BaseEstimator):

    def __init__(self, flags, category_index=None, multi_gpu=False):
        super(Estimator, self).__init__(flags)
        self.category_index = {1: {'id': 1, 'name': 'object'}}
        if category_index:
            self.category_index = category_index
        self.multi_gpu = multi_gpu

    def optimizer_fn(self, learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        return optimizer

    def new_model(self, features, labels, mode, params):
        raise NotImplementedError

    def detection_hook(self, features, labels, mode,
            prediction_dict,
            detection_dict):
        #visualization
        #image_with_box = tf.image.draw_bounding_boxes(features['images'], detection_dict['detection_boxes'])
        #tf.summary.image('predict_images', image_with_box)
        eval_dict = {
            'images': features['resized_images'],
            'groundtruth_boxes': labels['groundtruth_boxes'],
            'groundtruth_classes': labels['groundtruth_classes'],
            'detection_boxes': detection_dict['detection_boxes'],
            'detection_classes': detection_dict['detection_classes'],
            'detection_scores': detection_dict['detection_scores'],
            'num_detections': detection_dict['num_detections'],
        }
        draw_image_list = (
            vis_utils.draw_side_by_side_evaluation_images(
                eval_dict, self.category_index,
                max_images=3,
                max_boxes_to_draw=3,
                min_score_thresh=0.1))
        eval_image_summary = {}
        for i, draw_image in enumerate(draw_image_list):
            name = 'Detections_Left_Groundtruth_Right_{}'.format(i)
            image_summary = tf.summary.image(name,
                             draw_image)
            eval_image_summary[name] = (image_summary, tf.no_op())
        return eval_image_summary

    def model_fn(self, features, labels, mode, params):

        model = self.new_model(features, labels, mode, params)

        tf.logging.info('mode: %s', mode)
        prediction_dict, detection_dict = model(features, mode == tf.estimator.ModeKeys.TRAIN)

        predictions = detection_dict

        eval_image_summary = self.detection_hook(features, labels, mode, prediction_dict, detection_dict)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # Return the predictions and the specification for serving a SavedModel
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })

        tensors_to_log = model.tensors_to_log

        #loss
        loss_dict = model.loss(prediction_dict, labels)

        #model losses
        for loss_tensor in loss_dict.values():
            tf.losses.add_loss(loss_tensor)
            tf.summary.scalar(loss_tensor.op.name, loss_tensor)
            tensors_to_log[loss_tensor.op.name] = loss_tensor

        #regularization loss
        reg_loss = tf.losses.get_regularization_loss()
        loss_dict['Loss/regularization_loss'] = reg_loss
        tf.summary.scalar('Loss/regularization_loss', reg_loss)
        tensors_to_log['Loss/regularization_loss'] = reg_loss

        #total loss
        loss = tf.losses.get_total_loss()
        loss_dict['Loss/total_loss'] = loss
        tf.summary.scalar('Loss/total_loss', loss)
        tensors_to_log['Loss/total_loss'] = loss

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
            #train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)
            eval_metric_ops = None
        else:
            train_op = None
            #TODO: add eval
            eval_metric_ops = {}
            for loss_key, loss_tensor in iter(loss_dict.items()):
                eval_metric_ops[loss_key] = tf.metrics.mean(loss_tensor)
            #eval_metric_ops.update(eval_image_summary)

        training_hooks = []
        training_hooks.append(tf.train.LoggingTensorHook(
            tensors=tensors_to_log,
            every_n_iter=1))

        evaluation_hooks = []
        # Create a SummarySaverHook
        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir= self.flags.model_dir + "/eval",
            summary_op=tf.summary.merge_all())
        # Add it to the evaluation_hook list
        evaluation_hooks.append(eval_summary_hook)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            training_hooks=training_hooks,
            evaluation_hooks=evaluation_hooks,
            eval_metric_ops=eval_metric_ops)

