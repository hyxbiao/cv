from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf  # pylint: disable=g-bad-import-order
from imgcv import resnet
from imgcv.utils import preprocess as pp

_RESIZE_MIN = 256

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]


HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
'''
DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
RECORD_BYTES = DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 6

NUM_IMAGES = {
    'train': 10110,
    'validation': 1153,
}
'''
_SHUFFLE_BUFFER = 1500


class FashionAIDataSet(resnet.DataSet):
    CSV_TYPES = [[''], [''], ['']]
    CSV_COLUMN_NAMES = ['image', 'key', 'value']

    def __init__(self, flags):
        super(FashionAIDataSet, self).__init__(flags)

        df = self.load_meta_data(tf.estimator.ModeKeys.TRAIN)
        self._num_classes = len(df['value'].value_counts())
        self.train_df = df.sample(frac=0.9, random_state=1)
        self.test_df = df.drop(self.train_df.index)

        #self.train_label = self.train_df.pop('value')
        #self.test_label = self.test_df.pop('value')

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
        metas = self.get_metas(mode, self.flags.data_dir)
        dfs = []
        for data_dir, meta_file in metas:
            if convert:
                converters = {
                    'image': lambda x, data_dir=data_dir: os.path.join(data_dir, *x.split('/')),
                }
                if mode != tf.estimator.ModeKeys.PREDICT:
                    converters['value'] = lambda x: x.index('y')
            else:
                converters = None
            df = pd.read_csv(meta_file, names=self.CSV_COLUMN_NAMES, header=0, converters=converters)
            df = df[df['key'] == self.flags.attr_key]
            dfs.append(df)
        df = pd.concat(dfs)
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
        #is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        #examples_per_epoch = is_training and NUM_IMAGES['train'] or NUM_IMAGES['validation']
        shuffle_buffer = _SHUFFLE_BUFFER

        df = self.get_raw_input(mode)
        examples_per_epoch = len(df)

        dataset = tf.data.Dataset.from_tensor_slices(dict(df))
        if mode == tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.map(lambda value: self.parse_record(mode, value),
                                num_parallel_calls=self.num_parallel_calls)
        else:
            if mode == tf.estimator.ModeKeys.TRAIN:
                dataset = dataset.shuffle(buffer_size=df.shape[0])

            dataset = self.process(dataset, mode, shuffle_buffer, num_epochs, examples_per_epoch)
        return dataset

    def get_raw_input(self, mode, convert=True):
        if mode == tf.estimator.ModeKeys.TRAIN:
            df = self.train_df
        elif mode == tf.estimator.ModeKeys.EVAL:
            df = self.test_df
        elif mode == tf.estimator.ModeKeys.PREDICT:
            df = self.load_meta_data(mode, convert)

        return df

    def get_metas(self, mode, data_dir):
        """Returns a list of filenames."""
        metas = []
        if mode == tf.estimator.ModeKeys.PREDICT:
            rank_data_dir = os.path.join(data_dir, 'z_rank')
            metas.append((rank_data_dir, os.path.join(rank_data_dir, 'Tests', 'question.csv')))
        else:
            base_data_dir = os.path.join(data_dir, 'base')
            metas.append((base_data_dir, os.path.join(base_data_dir, 'Annotations', 'label.csv')))

            web_data_dir = os.path.join(data_dir, 'web')
            metas.append((web_data_dir, os.path.join(web_data_dir, 'Annotations', 'skirt_length_labels.csv')))
        return metas

    '''
    def parse_csv_record(self, mode, value):
        label = tf.cast(label, dtype=tf.int32)
        image_buffer = tf.read_file(meta['image'])
        if self.flags.debug:
            return {'path': meta['image'], 'image': image_buffer, 'label': label}
        return {'image': image_buffer, 'label': label}
    '''

    def parse_record(self, mode, record):
        image_buffer = tf.read_file(record['image'])

        if mode == tf.estimator.ModeKeys.PREDICT:
            image = self.preprocess_predict_image(mode, image_buffer)
            if self.flags.debug:
                return image, record['image']
            return image

        image = self.preprocess_image(mode, image_buffer)

        if self.flags.debug:
            return image, record['image'], record['value']

        label = record['value']
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


class FashionAIModel(resnet.Model):

    def __init__(self, resnet_size, data_format=None, num_classes=0,
           version=resnet.Model.DEFAULT_VERSION):
        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
            final_size = 512
        else:
            bottleneck = True
            final_size = 2048

        super(FashionAIModel, self).__init__(
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

    def get_block_sizes(self, resnet_size):
        """Retrieve the size of each block_layer in the ResNet model.

        The number of block layers used for the Resnet model varies according
        to the size of the model. This helper grabs the layer set we want, throwing
        an error if a non-standard size has been selected.

        Args:
            resnet_size: The number of convolutional layers needed in the model.

        Returns:
            A list of block sizes to use in building the model.

        Raises:
            KeyError: if invalid resnet_size is received.
        """
        choices = {
          18: [2, 2, 2, 2],
          34: [3, 4, 6, 3],
          50: [3, 4, 6, 3],
          101: [3, 4, 23, 3],
          152: [3, 8, 36, 3],
          200: [3, 24, 36, 3]
        }

        try:
            return choices[resnet_size]
        except KeyError:
            err = ('Could not find layers for selected Resnet size.\n'
                   'Size received: {}; sizes allowed: {}.'.format(
                       resnet_size, choices.keys()))
            raise ValueError(err)


class FashionAIEstimator(resnet.Estimator):
    def __init__(self, flags, train_num_images, num_classes):
        super(FashionAIEstimator, self).__init__(flags, model_class=FashionAIModel, weight_decay=1e-4)

        self.train_num_images = train_num_images
        self.num_classes = num_classes
        self.batch_size = flags.batch_size

    def new_model(self, features, labels, mode, params):
        resnet_size=params['resnet_size']
        data_format=params['data_format']
        version=params['version']
        model = self.model_class(resnet_size, data_format, num_classes=self.num_classes, version=version)
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
        tf.logging.info(labels)
        return super(FashionAIEstimator, self).model_fn(features, labels, mode, params)


class FashionAIRunner(resnet.Runner):
    def __init__(self, flags, estimator, dataset):
        shape = None
        super(FashionAIRunner, self).__init__(flags, estimator, dataset, shape)

    def run(self):
        '''
        config = tf.ConfigProto()
        if self.flags.gpu_allow_growth:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = self.flags.gpu_memory_fraction
        session = tf.Session(config=config)
        '''

        if self._run_debug():
            tf.logging.info('run debug finish')
            return
        output = super(FashionAIRunner, self).run()
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

        df = self.dataset.train_df
        print(len(df[df['value'] == 4]))
        print(self.dataset.train_df[:5])
        print(self.dataset.test_df[:5])
        print(self.dataset.train_num_images)
        print(self.dataset.test_num_images)
        print(self.dataset.num_classes)
        return True

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


class FashionAIArgParser(resnet.ArgParser):
    def __init__(self):
        super(FashionAIArgParser, self).__init__()
        self.add_argument(
            '--debug', '-dg', action='store_true',
            default=False,
            help='Debug'
        )
        self.add_argument(
            '--attr_key', '-ak',
            default='skirt_length_labels',
            choices=["skirt_length_labels", "neckline_design_labels", "collar_design_labels",
                "sleeve_length_labels", "neck_design_labels", "coat_length_labels",
                "lapel_design_labels", "pant_length_labels"],
            help='[default: %(default)s] Attribute key'
        )


def main(argv):
    parser = FashionAIArgParser()
    parser.set_defaults(data_dir='/tmp/data/fashionAI',
                        model_dir='./model_skirt_pretrain',
                        train_epochs=30,
                        predict_yield_single=False,
                        pretrain_warm_vars='^((?!dense).)*$')

    flags = parser.parse_args(args=argv[1:])

    dataset = FashionAIDataSet(flags)

    estimator = FashionAIEstimator(flags, 
            train_num_images=dataset.train_num_images,
            num_classes=dataset.num_classes)

    runner = FashionAIRunner(flags, estimator, dataset)
    runner.run()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    #log = logging.getLogger('tensorflow')
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #log.setFormatter(formatter)

    main(argv=sys.argv)
