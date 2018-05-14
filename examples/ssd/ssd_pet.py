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

NUM_CLASSES = 37

LABEL_NAMES = [
    'Abyssinian',
    'american_bulldog',
    'american_pit_bull_terrier',
    'basset_hound',
    'beagle',
    'Bengal',
    'Birman',
    'Bombay',
    'boxer',
    'British_Shorthair',
    'chihuahua',
    'Egyptian_Mau',
    'english_cocker_spaniel',
    'english_setter',
    'german_shorthaired',
    'great_pyrenees',
    'havanese',
    'japanese_chin',
    'keeshond',
    'leonberger',
    'Maine_Coon',
    'miniature_pinscher',
    'newfoundland',
    'Persian',
    'pomeranian',
    'pug',
    'Ragdoll',
    'Russian_Blue',
    'saint_bernard',
    'samoyed',
    'scottish_terrier',
    'shiba_inu',
    'Siamese',
    'Sphynx',
    'staffordshire_bull_terrier',
    'wheaten_terrier',
    'yorkshire_terrier',
]


class PetDataSet(ssd_common.SSDDataSet):
    def __init__(self, flags):
        super(PetDataSet, self).__init__(flags, NUM_CLASSES)

    def get_raw_dataset(self, mode, num_epochs=1):
        data_dir = os.path.expanduser(self.flags.data_dir)
        filenames = [
            os.path.join(data_dir, self.flags.train_file),
        ]
        if mode == tf.estimator.ModeKeys.PREDICT:
            filenames = [
                os.path.join(data_dir, self.flags.test_file),
            ]
        dataset = self.read_and_parse_dataset(filenames, num_epochs)
        return dataset


class PetEstimator(ssd_common.SSDEstimator):
    def __init__(self, flags, num_classes):
        category_index = {}
        for i, name in enumerate(LABEL_NAMES):
            index = i + 1
            category_index[index] = {
                'id': index,
                'name': name
            }
        super(PetEstimator, self).__init__(flags, num_classes, category_index)


class PetRunner(ssd_common.SSDRunner):
    def __init__(self, flags, estimator, dataset):
        super(PetRunner, self).__init__(flags, estimator, dataset)


def main(argv):

    parser = ssd_common.SSDArgParser()
    parser.set_defaults(data_dir='~/data/vision/Oxford-IIIT-Pet/tfrecords',
                        model_dir='./models/Oxford-IIIT-Pet/experiment',
                        train_file='pet_train_with_masks.record',
                        test_file='pet_val_with_masks.record',
                        )

    flags = parser.parse_args(args=argv[1:])
    tf.logging.info('flags: %s', flags)

    estimator = PetEstimator(flags, NUM_CLASSES)
    dataset = PetDataSet(flags)

    runner = PetRunner(flags, estimator, dataset)
    runner.run()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
