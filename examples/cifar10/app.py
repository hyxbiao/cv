from __future__ import print_function

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order
from imgcv.classification import resnet

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
RECORD_BYTES = DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 10
NUM_DATA_FILES = 5

NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


class Cifar10DataSet(resnet.DataSet):
    def __init__(self, flags):
        super(Cifar10DataSet, self).__init__(flags)

    def synth_input_fn(self, mode, num_epochs=1):
        images = tf.zeros((self.batch_size, HEIGHT, WIDTH, NUM_CHANNELS), tf.float32)
        labels = tf.zeros((self.batch_size, NUM_CLASSES), tf.int32)
        return tf.data.Dataset.from_tensors((images, labels)).repeat()

    def input_fn(self, mode, num_epochs=1):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        num_images = is_training and NUM_IMAGES['train'] or NUM_IMAGES['validation']
        shuffle_buffer = NUM_IMAGES['train']

        dataset = self.get_raw_input(mode)
        dataset = self.process(dataset, mode, shuffle_buffer, num_epochs, num_images)
        return dataset

    def get_raw_input(self, mode):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        filenames = self.get_filenames(is_training, self.flags.data_dir)
        dataset = tf.data.FixedLengthRecordDataset(filenames, RECORD_BYTES)

        return dataset

    def get_filenames(self, is_training, data_dir):
        """Returns a list of filenames."""
        data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

        assert os.path.exists(data_dir), (
            'Run cifar10_download_and_extract.py first to download and extract the '
            'CIFAR-10 data.')

        if is_training:
            return [
                os.path.join(data_dir, 'data_batch_%d.bin' % i)
                for i in range(1, NUM_DATA_FILES + 1)
            ]
        else:
            return [os.path.join(data_dir, 'test_batch.bin')]

    def parse_record(self, mode, record):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # Convert bytes to a vector of uint8 that is record_bytes long.
        record_vector = tf.decode_raw(record, tf.uint8)

        # The first byte represents the label, which we convert from uint8 to int32
        # and then to one-hot.
        label = tf.cast(record_vector[0], tf.int32)
        label = tf.one_hot(label, NUM_CLASSES)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(record_vector[1:RECORD_BYTES],
                               [NUM_CHANNELS, HEIGHT, WIDTH])

        # Convert from [depth, height, width] to [height, width, depth], and cast as
        # float32.
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

        image = self.preprocess_image(image, is_training)

        return image, label

    def preprocess_image(self, image, is_training):
        """Preprocess a single image of layout [height, width, depth]."""
        if is_training:
            # Resize the image to add four extra pixels on each side.
            image = tf.image.resize_image_with_crop_or_pad(
                    image, HEIGHT + 8, WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)
        return image


class Cifar10Model(resnet.Model):

    def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
           version=resnet.Model.DEFAULT_VERSION):
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)

        num_blocks = (resnet_size - 2) // 6

        super(Cifar10Model, self).__init__(
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


class Cifar10Estimator(resnet.Estimator):
    def __init__(self, flags):
        super(Cifar10Estimator, self).__init__(flags, model_class=Cifar10Model, weight_decay=2e-4)

        self.batch_size = flags.batch_size

    def loss_filter_fn(self, name):
        return True

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
        return super(Cifar10Estimator, self).model_fn(features, labels, mode, params)


def main(argv):
    parser = resnet.ArgParser()
    parser.set_defaults(data_dir='/tmp/cifar10_data',
                        model_dir='./model_cifar10',
                        resnet_size=32,
                        train_epochs=250,
                        epochs_between_evals=10,
                        batch_size=128)

    flags = parser.parse_args(args=argv[1:])

    estimator = Cifar10Estimator(flags)

    dataset = Cifar10DataSet(flags)


    '''
    mode = tf.estimator.ModeKeys.TRAIN
    ds = dataset.input_fn(mode, 1)
    #data = ds.make_one_shot_iterator().get_next()
    #with tf.Session() as sess:
    #    data = sess.run([data])
    #print(data)
    #return
    image, label = ds.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        image, label = sess.run([image, label])

    print('image')
    print(image)
    #features = tf.reshape(features, [-1, HEIGHT, WIDTH, NUM_CHANNELS])
    print(label)

    return
    '''

    runner = resnet.Runner(flags, estimator, dataset)
    #runner = resnet.Runner(flags, estimator.model_fn, dataset.synth_input_fn)
    runner.run()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
