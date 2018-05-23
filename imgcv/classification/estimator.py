#coding=utf-8
#date: 2018-04-12
#author: hyxbiao

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.estimator import BaseEstimator


class Estimator(BaseEstimator):

    def __init__(self, flags, learning_rate=0.1, weight_decay=1e-4, multi_gpu=False):
        super(Estimator, self).__init__(flags)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.multi_gpu = multi_gpu

    def learning_rate_fn(self, global_step):
        return tf.constant(self.learning_rate)

    def loss_filter_fn(self, name):
        return 'batch_normalization' not in name

    def optimizer_fn(self, learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        return optimizer

    def new_model(self, features, labels, mode, params):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params):
        # Generate a summary node for the images
        tf.summary.image('images', features, max_outputs=6)

        model = self.new_model(features, labels, mode, params)
        logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

        if self.flags.score_fn == 'sigmoid':
            score_fn = tf.sigmoid
            loss_fn = tf.losses.sigmoid_cross_entropy
        else:
            score_fn = tf.nn.softmax
            loss_fn = tf.losses.softmax_cross_entropy

        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': score_fn(logits, name='prob_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            # Return the predictions and the specification for serving a SavedModel
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })

        # Calculate loss, which includes softmax cross entropy and L2 regularization.
        cross_entropy = loss_fn(labels, logits)

        # Create a tensor named cross_entropy for logging purposes.
        tf.identity(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)

        # Add weight decay to the loss.
        l2_loss = self.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
             if self.loss_filter_fn(v.name)])
        tf.summary.scalar('l2_loss', l2_loss)
        loss = cross_entropy + l2_loss

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

        accuracy = tf.metrics.accuracy(
                tf.argmax(labels, axis=1), predictions['classes'])
        metrics = {'accuracy': accuracy}

        # Create a tensor named train_accuracy for logging purposes
        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

