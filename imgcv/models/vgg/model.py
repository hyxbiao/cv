from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

from imgcv.models.model import Model as BaseModel
from imgcv.models.vgg import vgg


class Model(BaseModel):
    """Base class for building the Vgg Model."""

    VGG_a   = 'vgg_a'
    VGG_16  = 'vgg_16'
    VGG_19  = 'vgg_19'

    def __init__(self, num_classes, mode=VGG_16,
           dropout_keep_prob=0.5):
        super(Model, self).__init__()

        self.mode = mode
        if mode == self.VGG_16:
            self.model = vgg.vgg_16
        elif mode == self.VGG_19:
            self.model = vgg.vgg_19
        elif mode == self.VGG_a:
            self.model = vgg.vgg_a
        else:
            raise ValueError('Not support vgg mode')

        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze=True
        self.scope = mode
        self.fc_conv_padding='VALID'
        self.global_pool=False

    def __call__(self, inputs, training):
        """Add operations to classify a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Set to True to add operations required only when
                training the classifier.

        Returns:
            A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        with slim.arg_scope(vgg.vgg_arg_scope()):
            net, end_points = self.model(inputs,
                        num_classes=self.num_classes,
                        is_training=training,
                        dropout_keep_prob=self.dropout_keep_prob,
                        spatial_squeeze=self.spatial_squeeze,
                        scope=self.scope,
                        fc_conv_padding=self.fc_conv_padding,
                        global_pool=self.global_pool)
        return net
