#coding=utf-8
#date: 2018-04-11
#author: hyxbiao

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order
from imgcv.dataset import EstimatorDataSet

class DataSet(EstimatorDataSet):

    def __init__(self, flags):
        super(DataSet, self).__init__(flags)
        self.batch_size = flags.batch_size
        self.multi_gpu = flags.multi_gpu 
        self.num_parallel_calls = flags.num_parallel_calls 

    def process(self, dataset, mode, shuffle_buffer, num_epochs, examples_per_epoch):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # We prefetch a batch at a time, This can help smooth out the time taken to
        # load input files as we go through shuffling and processing.
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        if is_training:
            # Shuffle the records. Note that we shuffle before repeating to ensure
            # that the shuffling respects epoch boundaries.
            dataset = dataset.shuffle(buffer_size=shuffle_buffer)

        # If we are training over multiple epochs before evaluating, repeat the
        # dataset for the appropriate number of epochs.
        dataset = dataset.repeat(num_epochs)

        # Currently, if we are using multiple GPUs, we can't pass in uneven batches.
        # (For example, if we have 4 GPUs, the number of examples in each batch
        # must be divisible by 4.) We already ensured this for the batch_size, but
        # we have to additionally ensure that any "leftover" examples-- the remainder
        # examples (total examples % batch_size) that get called a batch for the very
        # last batch of an epoch-- do not raise an error when we try to split them
        # over the GPUs. This will likely be handled by Estimator during replication
        # in the future, but for now, we just drop the leftovers here.
        if self.multi_gpu:
            total_examples = num_epochs * examples_per_epoch
            dataset = dataset.take(self.batch_size * (total_examples // self.batch_size))

        # Parse the raw records into images and labels
        dataset = dataset.map(lambda value: self.parse_record(mode, value),
                            num_parallel_calls=self.num_parallel_calls)

        dataset = dataset.batch(self.batch_size)

        # Operations between the final prefetch and the get_next call to the iterator
        # will happen synchronously during run time. We prefetch here again to
        # background all of the above processing work and keep it out of the
        # critical training path.
        dataset = dataset.prefetch(1)

        return dataset
