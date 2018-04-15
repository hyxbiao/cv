from __future__ import print_function

import os
import sys

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
DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
RECORD_BYTES = DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 6
NUM_DATA_FILES = 5

NUM_IMAGES = {
    'train': 10110,
    'validation': 1153,
}
_SHUFFLE_BUFFER = 1500


class FashionAIDataSet(resnet.DataSet):
    CSV_TYPES = [[''], [''], ['']]
    CSV_COLUMN_NAMES = ['image', 'key', 'value']

    def __init__(self, flags):
        super(FashionAIDataSet, self).__init__(flags)

    def debug_fn(self):
        mode = tf.estimator.ModeKeys.TRAIN
        dataset = self.get_raw_input(mode)
        #dataset = self.input_fn(mode)
        dataset = dataset.map(lambda value: self.parse_record(mode, value),
                            num_parallel_calls=self.num_parallel_calls)
        return dataset

    def input_fn(self, mode, num_epochs=1):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        num_images = is_training and NUM_IMAGES['train'] or NUM_IMAGES['validation']
        shuffle_buffer = _SHUFFLE_BUFFER

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
        if is_training:
            dataset = dataset.shuffle(buffer_size=df.shape[0])

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
        image_buffer = tf.read_file(meta['image'])
        if self.flags.debug:
            return {'path': meta['image'], 'image': image_buffer, 'label': label}
        return {'image': image_buffer, 'label': label}

    def parse_record(self, mode, record):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # and then to one-hot.
        label = record['label']
        label = tf.one_hot(label, NUM_CLASSES)

        image_buffer = record['image']
        image = self.preprocess_image(image_buffer, is_training)

        if self.flags.debug:
            return image, record['label'], record['path']
        return image, label

    def preprocess_image(self, image_buffer, is_training):
        if is_training:
            raw_image = tf.image.decode_jpeg(image_buffer, channels=NUM_CHANNELS)
            small_image = pp.image.aspect_preserving_resize(raw_image, _RESIZE_MIN)
            #crop_image = tf.random_crop(small_image, [HEIGHT, WIDTH, NUM_CHANNELS])
            crop_image = pp.image.central_crop(small_image, HEIGHT, WIDTH, vertical=pp.image.VERTICAL_TOP)
            image = tf.image.random_flip_left_right(crop_image)
        else:
            raw_image = tf.image.decode_jpeg(image_buffer, channels=NUM_CHANNELS)
            small_image = pp.image.aspect_preserving_resize(raw_image, _RESIZE_MIN)
            #image = pp.image.central_crop(small_image, HEIGHT, WIDTH)
            image = pp.image.central_crop(small_image, HEIGHT, WIDTH, vertical=pp.image.VERTICAL_TOP)
            
        image.set_shape([HEIGHT, WIDTH, NUM_CHANNELS])

        image = pp.image.mean_image_subtraction(image, _CHANNEL_MEANS, NUM_CHANNELS)
        if self.flags.debug:
            return raw_image, small_image, image
        return image

    def preprocess_image_v1(self, image, is_training):
        """Preprocess a single image of layout [height, width, depth]."""
        raw_image = tf.image.decode_jpeg(image, channels=NUM_CHANNELS)
        if is_training:
            # Resize the image to add four extra pixels on each side.
            resize_image = tf.image.resize_image_with_crop_or_pad(
                    raw_image, HEIGHT + 8, WIDTH + 8)
        else:
            resize_image = tf.image.resize_images(raw_image, (HEIGHT, WIDTH))

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        crop_image = tf.random_crop(resize_image, [HEIGHT, WIDTH, NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(crop_image)

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)
        if self.flags.debug:
            return raw_image, resize_image, crop_image, image
        return image


class FashionAIModel(resnet.Model):

    def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
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
    def __init__(self, flags):
        super(FashionAIEstimator, self).__init__(flags, model_class=FashionAIModel, weight_decay=1e-4)

        self.batch_size = flags.batch_size

    def optimizer_fn(self, learning_rate):
        optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.9)
        return optimizer

    def learning_rate_fn(self, global_step):
        batch_denom = 256
        num_images = NUM_IMAGES['train']
        boundary_epochs = [30, 60, 80, 90]
        decay_rates = [1, 0.1, 0.01, 0.001, 1e-4]

        initial_learning_rate = 0.1 * self.batch_size / batch_denom
        batches_per_epoch = num_images / self.batch_size

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
        vals = [initial_learning_rate * decay for decay in decay_rates]

        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    def model_fn(self, features, labels, mode, params):
        return super(FashionAIEstimator, self).model_fn(features, labels, mode, params)


class FashionAIRunner(resnet.Runner):
    def __init__(self, flags, estimator, dataset):
        shape = None
        super(FashionAIRunner, self).__init__(flags, estimator, dataset, shape)

    def run(self):
        if self._run_debug():
            tf.logging.info('run debug finish')
            return
        super(FashionAIRunner, self).run()

    def _run_debug(self):
        if not self.flags.debug:
            return False

        ds = self.dataset.debug_fn()
        data = ds.make_one_shot_iterator().get_next()

        tf.logging.info(data)

        writer = tf.summary.FileWriter('./debug')
        
        '''
        for i in range(len(data[0])):
            name = 'image{}'.format(i)
            #name = 'image'
            family = 'image'
            image = tf.expand_dims(data[0][i], 0)
            summary_op = tf.summary.image(name, image, max_outputs=3, family=family)
            #summary_op = tf.summary.image(name, image, max_outputs=3)
        #tf.summary.image('image', tf.stack(data[0]), max_outputs=3)
        tf.summary.text("path", data[-1])

        summary_op = tf.summary.merge_all()
        '''

        with tf.Session() as sess:
            for step in range(10):
                #label = sess.run(data[-2])
                #path = sess.run(data[-1])
                sops = []
                image_count = len(data[0])
                for i in range(image_count):
                    #name = '{}_{}_{}'.format(i, label, path)
                    name = '{}'.format(i)
                    family = 'step{}'.format(step)
                    image = tf.expand_dims(data[0][i], 0)
                    sop = tf.summary.image(name, image, max_outputs=image_count, family=family)
                    sops.append(sop)
                sop = tf.summary.scalar("label", data[-2])
                sops.append(sop)
                sop = tf.summary.text("path", data[-1])
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



def main(argv):
    parser = FashionAIArgParser()
    parser.set_defaults(data_dir='/tmp/data/fashionAI',
                        model_dir='./model_skirt_pretrain',
                        train_epochs=30,
                        pretrain_warm_vars='^((?!dense).)*$')

    flags = parser.parse_args(args=argv[1:])

    estimator = FashionAIEstimator(flags)

    dataset = FashionAIDataSet(flags)

    runner = FashionAIRunner(flags, estimator, dataset)
    runner.run()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
