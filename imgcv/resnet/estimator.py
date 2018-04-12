#coding=utf-8
#date: 2018-04-12
#author: hyxbiao

import os
import sys

from imgcv.estimator import ClassifyEstimator
from imgcv.resnet import Model

class Estimator(ClassifyEstimator):

    def __init__(self, flags, model_class=Model, **kwargs):
        super(Estimator, self).__init__(flags, kwargs)
        self.model_class = model_class

    def new_model(self, features, labels, mode, params):
        resnet_size=params['resnet_size']
        data_format=params['data_format']
        version=params['version']
        model = self.model_class(resnet_size, data_format, version=version)
        return model

