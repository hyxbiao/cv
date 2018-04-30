#coding=utf-8
#date: 2018-04-11
#author: hyxbiao

import os
import sys

class Model(object):

    def __init__(self):
        pass

    def __call__(self, inputs, training):
        raise NotImplementedError
