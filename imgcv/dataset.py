#coding=utf-8
#date: 2018-04-11
#author: hyxbiao

import os
import sys

class BaseDataSet(object):

    def __init__(self, flags):
        self.flags = flags


class EstimatorDataSet(BaseDataSet):

    def __init__(self, flags):
        super(EstimatorDataSet, self).__init__(flags)

    def input_fn(self, mode, num_epochs=1):
        raise NotImplementedError

    def parse_record(self, mode, record):
        raise NotImplementedError

