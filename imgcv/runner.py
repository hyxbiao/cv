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
        raise NotImplementedError
