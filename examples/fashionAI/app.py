from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf  # pylint: disable=g-bad-import-order
from imgcv import resnet

HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
RECORD_BYTES = DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 6
NUM_DATA_FILES = 5

NUM_IMAGES = {
    'train': 10110,
    'validation': 1153,
}


class FashionAIDataSet(resnet.DataSet):
    CSV_TYPES = [[''], [''], ['']]
    CSV_COLUMN_NAMES = ['image', 'key', 'value']

    def __init__(self, flags):
        super(FashionAIDataSet, self).__init__(flags)

    def input_fn(self, mode, num_epochs=1):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        num_images = is_training and NUM_IMAGES['train'] or NUM_IMAGES['validation']
        shuffle_buffer = NUM_IMAGES['train']

        dataset = self.get_raw_input(mode)
        dataset = self.process(dataset, mode, shuffle_buffer, num_epochs, num_images)
        return dataset

    def get_raw_input(self, mode):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        data_dir, meta_file = self.get_filenames(is_training, self.flags.data_dir)

        if is_training:
            value_func = lambda x: x.index('y')
        else:
            value_func = lambda x: np.argmax(map(float, x.split(';')))
        converters = {
            'image': lambda x: os.path.join(data_dir, x),
            'value': value_func
        }
        df = pd.read_csv(meta_file, names=self.CSV_COLUMN_NAMES, header=0, converters=converters)
        df = df[df['key'] == 'skirt_length_labels']
        label = df.pop('value')
        dataset = tf.data.Dataset.from_tensor_slices((dict(df), label))

        dataset = dataset.map(lambda meta, label: self.parse_csv_record(meta, label, data_dir))
        #dataset = tf.data.FixedLengthRecordDataset(filenames, RECORD_BYTES)

        return dataset

    def get_filenames(self, is_training, data_dir):
        """Returns a list of filenames."""
        if is_training:
            data_dir = os.path.join(data_dir, 'web')
            return (data_dir, os.path.join(data_dir, 'Annotations/skirt_length_labels.csv'))
        else:
            data_dir = os.path.join(data_dir, 'rank')
            return (data_dir, os.path.join(data_dir, 'Tests/answer_mock.csv'))

    def parse_csv_record(self, meta, label, data_dir):
        label = tf.cast(label, dtype=tf.int32)
        image = tf.read_file(meta['image'])
        image = tf.image.decode_jpeg(image, NUM_CHANNELS)
        return {'image': image, 'label': label}

    def parse_record(self, mode, record):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # and then to one-hot.
        label = record['label']
        label = tf.one_hot(label, NUM_CLASSES)


        image = record['image']
        #image = tf.image.resize_images(image, (HEIGHT, WIDTH))
        image = tf.cast(image, dtype=tf.float32)

        image = self.preprocess_image(image, is_training)

        return image, label

    def preprocess_image(self, image, is_training):
        """Preprocess a single image of layout [height, width, depth]."""
        if is_training:
            # Resize the image to add four extra pixels on each side.
            image = tf.image.resize_image_with_crop_or_pad(
                    image, HEIGHT + 8, WIDTH + 8)
        else:
            image = tf.image.resize_images(image, (HEIGHT, WIDTH))

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)
        return image


class FashionAIModel(resnet.Model):

    def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
           version=resnet.Model.DEFAULT_VERSION):
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)

        num_blocks = (resnet_size - 2) // 6

        super(FashionAIModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=False,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            second_pool_size=8,
            second_pool_stride=1,
            block_sizes=[num_blocks] * 3,
            block_strides=[1, 2, 2],
            final_size=64,
            version=version,
            data_format=data_format)


class FashionAIEstimator(resnet.Estimator):
    def __init__(self, flags):
        super(FashionAIEstimator, self).__init__(flags, model_class=FashionAIModel, weight_decay=2e-4)

        self.batch_size = flags.batch_size

    def optimizer_fn(self, learning_rate):
        optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.9)
        return optimizer

    def learning_rate_fn(self, global_step):
        batch_denom = 128
        num_images = NUM_IMAGES['train']
        boundary_epochs = [100, 150, 200]
        decay_rates = [1, 0.1, 0.01, 0.001]

        initial_learning_rate = 0.1 * self.batch_size / batch_denom
        batches_per_epoch = num_images / self.batch_size

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
        vals = [initial_learning_rate * decay for decay in decay_rates]

        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    def model_fn(self, features, labels, mode, params):
        features = tf.reshape(features, [-1, HEIGHT, WIDTH, NUM_CHANNELS])
        return super(FashionAIEstimator, self).model_fn(features, labels, mode, params)


def test(dataset):
    mode = tf.estimator.ModeKeys.TRAIN
    ds = dataset.input_fn(mode, 1)
    data = ds.make_one_shot_iterator().get_next()
    print(data[0])
    print(data[1])
    return
    with tf.Session() as sess:
        data = sess.run([data])
    #print(data)

def main(argv):
    parser = resnet.ArgParser()
    parser.set_defaults(data_dir='/tmp/data/fashionAI',
                        model_dir='./model_skirt',
                        resnet_size=32,
                        train_epochs=250,
                        epochs_between_evals=10,
                        batch_size=128)

    flags = parser.parse_args(args=argv[1:])

    estimator = FashionAIEstimator(flags)

    dataset = FashionAIDataSet(flags)

    runner = resnet.Runner(flags, estimator.model_fn, dataset.input_fn)
    runner.run()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
