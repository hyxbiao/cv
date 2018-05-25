from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob

import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf  # pylint: disable=g-bad-import-order
from imgcv.dataset import DataSet
from imgcv import classification as cls
from imgcv.models import resnet
from imgcv.utils import preprocess as pp
from tornado import gen

_RESIZE_MIN = 256

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]


HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
_NUM_CLASSES = 6
_CLASSES = [
    'cloudy',
    'foggy',
    'rain',
    'snow',
    'sunny',
    'z-other',
]


class WeatherDataSet(DataSet):

    def __init__(self, flags):
        super(WeatherDataSet, self).__init__(flags)
        self.data_dir = os.path.expanduser(self.flags.data_dir)

        df = self.load_meta_data(tf.estimator.ModeKeys.TRAIN)
        self._num_classes = _NUM_CLASSES
        self.train_df = df.sample(frac=0.8, random_state=1)
        self.test_df = df.drop(self.train_df.index)
        tf.logging.info('train fiels: %d, test files: %d', len(self.train_df), len(self.test_df))

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def train_num_images(self):
        return len(self.train_df)

    @property
    def test_num_images(self):
        return len(self.test_df)

    def load_meta_data(self, mode, convert=True):
        pattern = '{}/*/*.jpg'.format(self.data_dir)
        filenames = glob.glob(pattern)

        def transfer(filename):
            name = filename.rsplit('/', 2)[1]
            label = _CLASSES.index(name)
            return {'image': filename, 'label': label}

        data = map(transfer, filenames)
        df = pd.DataFrame(data)
        return df

    def debug_fn(self):
        if self.flags.predict:
            mode = tf.estimator.ModeKeys.PREDICT
        else:
            mode = tf.estimator.ModeKeys.TRAIN
        df = self.get_raw_input(mode)
        dataset = tf.data.Dataset.from_tensor_slices(dict(df))
        #dataset = self.input_fn(mode)
        dataset = dataset.map(lambda value: self.parse_record(mode, value),
                            num_parallel_calls=self.num_parallel_calls)
        return dataset

    def input_fn(self, mode, num_epochs=1):

        df = self.get_raw_input(mode)
        examples_per_epoch = len(df)
        shuffle_buffer = len(df)

        dataset = tf.data.Dataset.from_tensor_slices(dict(df))
        if mode == tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.map(lambda value: self.parse_record(mode, value),
                                num_parallel_calls=self.num_parallel_calls)
        else:
            dataset = self.process(dataset, mode, shuffle_buffer, num_epochs, examples_per_epoch)
        return dataset

    def get_raw_input(self, mode, convert=True):
        if mode == tf.estimator.ModeKeys.TRAIN:
            df = self.train_df
        elif mode == tf.estimator.ModeKeys.EVAL:
            df = self.test_df
        elif mode == tf.estimator.ModeKeys.PREDICT:
            if self.flags.predict_input_file:
                df = pd.DataFrame(data={'image': [self.flags.predict_input_file]})
            else:
                df = self.load_meta_data(mode, convert)

        return df

    def parse_record(self, mode, record):
        image_buffer = tf.read_file(record['image'])

        if mode == tf.estimator.ModeKeys.PREDICT:
            image = self.preprocess_predict_image(mode, image_buffer)
            if self.flags.debug:
                return image, record['image']
            return image

        image = self.preprocess_image(mode, image_buffer)

        label = record['label']
        if self.flags.debug:
            return image, record['image'], label

        label = tf.one_hot(label, self._num_classes)

        return image, label

    def preprocess_image(self, mode, image_buffer):
        raw_image = tf.image.decode_jpeg(image_buffer, channels=NUM_CHANNELS)
        small_image = pp.image.aspect_preserving_resize(raw_image, _RESIZE_MIN)
        if mode == tf.estimator.ModeKeys.TRAIN:
            crop_image = tf.random_crop(small_image, [HEIGHT, WIDTH, NUM_CHANNELS])
            image = tf.image.random_flip_left_right(crop_image)
        elif mode == tf.estimator.ModeKeys.EVAL:
            image = pp.image.central_crop(small_image, HEIGHT, WIDTH)

        image.set_shape([HEIGHT, WIDTH, NUM_CHANNELS])

        #image = tf.image.per_image_standardization(image)
        image = pp.image.mean_image_subtraction(image, _CHANNEL_MEANS, NUM_CHANNELS)
        if self.flags.debug:
            return raw_image, small_image, image
        return image

    def preprocess_predict_image(self, mode, image_buffer):
        raw_image = tf.image.decode_jpeg(image_buffer, channels=NUM_CHANNELS)
        image = pp.image.aspect_preserving_resize(raw_image, _RESIZE_MIN)

        #image = tf.random_crop(small_image, [HEIGHT, WIDTH, NUM_CHANNELS])
        images = [
            pp.image.central_crop(image, HEIGHT, WIDTH),
            pp.image.top_left_crop(image, HEIGHT, WIDTH),
            pp.image.top_right_crop(image, HEIGHT, WIDTH),
            pp.image.bottom_left_crop(image, HEIGHT, WIDTH),
            pp.image.bottom_right_crop(image, HEIGHT, WIDTH),
        ]
        images += [tf.image.flip_left_right(image) for image in images]
        if self.flags.debug:
            return tuple(images)
        return tf.stack(images)


class WeatherModel(resnet.Model):

    def __init__(self, resnet_size, data_format=None, num_classes=0,
           version=resnet.Model.DEFAULT_VERSION):
        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
            final_size = 512
        else:
            bottleneck = True
            final_size = 2048

        super(WeatherModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            block_sizes=self.get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            final_size=final_size,
            version=version,
            data_format=data_format)


class WeatherEstimator(cls.Estimator):
    def __init__(self, flags, train_num_images, num_classes):
        super(WeatherEstimator, self).__init__(flags, weight_decay=1e-4)

        self.train_num_images = train_num_images
        self.num_classes = num_classes
        self.batch_size = flags.batch_size

    def new_model(self, features, labels, mode, params):
        resnet_size = self.flags.resnet_size
        data_format = self.flags.data_format
        model = WeatherModel(resnet_size, data_format, num_classes=self.num_classes)
        return model

    def optimizer_fn(self, learning_rate):
        optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.9)
        return optimizer

    def learning_rate_fn(self, global_step):
        batch_denom = 256
        boundary_epochs = [30, 60, 80, 90]
        decay_rates = [1, 0.1, 0.01, 0.001, 1e-4]

        initial_learning_rate = 0.1 * self.batch_size / batch_denom
        batches_per_epoch = self.train_num_images / self.batch_size

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
        vals = [initial_learning_rate * decay for decay in decay_rates]

        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    def model_fn(self, features, labels, mode, params):
        tf.logging.info(features)
        return super(WeatherEstimator, self).model_fn(features, labels, mode, params)


class WeatherRunner(cls.Runner):
    def __init__(self, flags, estimator, dataset):
        shape = None
        super(WeatherRunner, self).__init__(flags, estimator, dataset, shape)

    def run(self):
        if self._run_debug():
            tf.logging.info('run debug finish')
            return
        output = super(WeatherRunner, self).run()
        if self.flags.predict:
            self.process_predict_output(output)

    def process_predict_output(self, output):
        df = self.dataset.get_raw_input(tf.estimator.ModeKeys.PREDICT, convert=False)
        tf.logging.info('total count: %d', len(df))
        writer = None
        if self.flags.predict_output_dir:
            filename = os.path.join(self.flags.predict_output_dir, 'output.csv')
            writer = open(filename, 'w')
        for i, v in enumerate(output):
            r = df.iloc[i]
            prob = np.mean(v['probabilities'], axis=0)
            pred_label = np.argmax(prob)
            prob_str = ';'.join(np.char.mod('%.4f', prob))
            r['value'] = prob_str
            tf.logging.info('[%d] image: %s, prob: %s, pred: %d', i, r['image'], prob_str, pred_label)
            if writer:
                writer.write('{},{},{}\n'.format(r['image'], r['key'], r['value']))
                writer.flush()
        if writer:
            writer.close()
        tf.logging.info('write to file done!')

    def _run_debug(self):
        if not self.flags.debug:
            return False

        ds = self.dataset.debug_fn()
        data = ds.make_one_shot_iterator().get_next()

        tf.logging.info(data)

        writer = tf.summary.FileWriter('./debug')

        with tf.Session() as sess:
            for step in range(10):
                sops = []
                image_count = len(data[0])
                for i in range(image_count):
                    #name = '{}_{}_{}'.format(i, label, path)
                    name = '{}'.format(i)
                    family = 'step{}'.format(step)
                    image = tf.expand_dims(data[0][i], 0)
                    sop = tf.summary.image(name, image, max_outputs=image_count, family=family)
                    sops.append(sop)
                if len(data) >= 2:
                    sop = tf.summary.text("path", data[1])
                    sops.append(sop)
                if len(data) >= 3:
                    sop = tf.summary.scalar("label", data[2])
                    sops.append(sop)
                summary_op = tf.summary.merge(sops)
                summary = sess.run(summary_op)
                writer.add_summary(summary, step)
                step += 1

        writer.close()
        return True


class WeatherArgParser(cls.ArgParser):
    def __init__(self, resnet_size_choices=None):
        super(WeatherArgParser, self).__init__()
        self.add_argument(
            '--version', '-v', type=int, choices=[1, 2],
            default=resnet.Model.DEFAULT_VERSION,
            help='Version of ResNet. (1 or 2) See README.md for details.'
        )

        self.add_argument(
            '--resnet_size', '-rs', type=int, default=50,
            choices=resnet_size_choices,
            help='[default: %(default)s] The size of the ResNet model to use.',
            metavar='<RS>' if resnet_size_choices is None else None
        )
        self.add_argument(
            '--debug', '-dg', action='store_true',
            default=False,
            help='Debug'
        )
        self.add_argument(
            "--predict_input_file",
            help="Predict input file",
        )
        self.add_argument(
            '--display', action='store_true',
            default=False,
            help='Display'
        )


def main(argv):
    parser = WeatherArgParser()
    parser.set_defaults(data_dir='~/data/vision/weather/Image',
                        model_dir='./models/experiment',
                        train_epochs=10,
                        predict_yield_single=False,
                        pretrain_model_dir='./pretrain_resnet50',
                        pretrain_warm_vars='^((?!dense).)*$')

    flags = parser.parse_args(args=argv[1:])

    dataset = WeatherDataSet(flags)

    estimator = WeatherEstimator(flags,
            train_num_images=dataset.train_num_images,
            num_classes=dataset.num_classes)

    runner = WeatherRunner(flags, estimator, dataset)
    runner.run()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
