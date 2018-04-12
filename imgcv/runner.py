#coding=utf-8
#date: 2018-04-11
#author: hyxbiao

import os
import sys

class BaseRunner(object):

    def __init__(self, flags):
        self.flags = flags

    def run(self):
        raise NotImplementedError


class EstimatorRunner(BaseRunner):

    def __init__(self, flags, model_function, input_function):
        super(EstimatorRunner, self).__init__(flags)
        self.model_function = model_function
        self.input_function = input_function

    def run(self):
        raise NotImplementedError
