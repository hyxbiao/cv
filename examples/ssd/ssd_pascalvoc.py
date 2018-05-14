from __future__ import print_function

import os
import sys
import collections
import functools

import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.dataset import DetectionDataSet
from imgcv.dataset import batcher
from imgcv import detection
from imgcv.models import ssd
from imgcv.models import vgg
from imgcv.utils import preprocess as pp
from imgcv.utils import ops

import ssd_common

slim = tf.contrib.slim

NUM_CLASSES = 20
NUM_IMAGES = {
    'train': 5011,
    'test': 4952,
}

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}


class PascalVocDataSet(ssd_common.SSDDataSet):
    def __init__(self, flags):
        super(PascalVocDataSet, self).__init__(flags, NUM_CLASSES)

    def get_raw_dataset(self, mode, num_epochs=1):
        data_dir = os.path.expanduser(self.flags.data_dir)
        filenames = [
            os.path.join(data_dir, 'trainval_merge.record'),
        ]
        if mode == tf.estimator.ModeKeys.PREDICT:
            filenames = [
                os.path.join(data_dir, 'test.record'),
            ]
        dataset = self.read_and_parse_dataset(filenames, num_epochs)
        return dataset


class PascalVocEstimator(ssd_common.SSDEstimator):
    def __init__(self, flags, num_classes):
        category_index = {}
        for name in VOC_LABELS:
            index = VOC_LABELS[name][0]
            category_index[index] = {
                'id': index,
                'name': name
            }
        super(PascalVocEstimator, self).__init__(flags, num_classes, category_index)


class PascalVocRunner(ssd_common.SSDRunner):
    def __init__(self, flags, estimator, dataset):
        super(PascalVocRunner, self).__init__(flags, estimator, dataset)


def main(argv):

    parser = ssd_common.SSDArgParser()
    parser.set_defaults(data_dir='~/data/vision/pascal-voc/tfrecords/',
                        model_dir='./models/pascal-voc/experiment',
                        )

    flags = parser.parse_args(args=argv[1:])
    tf.logging.info('flags: %s', flags)

    estimator = PascalVocEstimator(flags, NUM_CLASSES)
    dataset = PascalVocDataSet(flags)

    runner = PascalVocRunner(flags, estimator, dataset)
    runner.run()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
