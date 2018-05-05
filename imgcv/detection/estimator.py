#coding=utf-8
#date: 2018-05-03
#author: hyxbiao

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.estimator import BaseEstimator


class Estimator(BaseEstimator):

    def __init__(self, flags,
            learning_rate=0.1, weight_decay=1e-4, multi_gpu=False):
        super(Estimator, self).__init__(flags)
        self.weight_decay = weight_decay
        self.multi_gpu = multi_gpu

    def optimizer_fn(self, learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        return optimizer

    def new_model(self, features, labels, mode, params):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params):

        model = self.new_model(features, labels, mode, params)

        resized_inputs, boxes_list, classes_list = model.get_inputs(features)
        prediction_dict = model(resized_inputs, mode == tf.estimator.ModeKeys.TRAIN)

        predictions = {
            'box_encodings': prediction_dict['box_encodings'],
            'class_predictions_with_background':
                prediction_dict['class_predictions_with_background'],
        }
        sys.exit(0)

        if mode == tf.estimator.ModeKeys.PREDICT:
            #postprocess
            #outputs

            # Return the predictions and the specification for serving a SavedModel
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })

        loss_dict = model.loss(prediction_dict, boxes_list, classes_list)
        for loss_tensor in loss_dict.values():
            tf.losses.add_loss(loss_tensor)
        loss = tf.losses.get_total_loss()

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

        #TODO: add eval
        metrics = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

