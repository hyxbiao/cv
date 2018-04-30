#coding=utf-8
#date: 2018-04-12
#author: hyxbiao

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

class BaseEstimator(object):

    def __init__(self, flags):
        self.flags = flags

    def model_fn(self, features, labels, mode, params):
        raise NotImplementedError

