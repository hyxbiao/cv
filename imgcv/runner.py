#coding=utf-8
#date: 2018-04-11
#author: hyxbiao

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

class BaseRunner(object):

    def __init__(self, flags):
        self.flags = flags

    def run(self):
        raise NotImplementedError


class EstimatorRunner(BaseRunner):

    def __init__(self, flags, estimator, dataset):
        super(EstimatorRunner, self).__init__(flags)
        self.estimator = estimator
        self.dataset = dataset

    def run(self):
        if self._run_hook():
            tf.logging.info('run hook finish')
            return
        self.run_internal()

    def run_internal(self):
        raise NotImplementedError

    def _run_hook(self):
        if 'debug_dataset' not in self.flags:
            return False

        ds = self.dataset.debug_fn()
        if ds is False:
            tf.logging.warning('Need define debug_fn')
            return True
        if self.flags.debug_dataset == 'one_shot':
            data = ds.make_one_shot_iterator().get_next()
        elif self.flags.debug_dataset == 'batch':
            data = ds.make_one_shot_iterator().get_next()
        else:
            tf.logging.warning('No match dataset choice')
            return True

        tf.logging.info(data)

        #with tf.Session() as sess:
        #    data = sess.run([data])
        #tf.logging.info(data)
        return True

