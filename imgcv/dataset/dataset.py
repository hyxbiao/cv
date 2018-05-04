#coding=utf-8
#date: 2018-04-11
#author: hyxbiao

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order
from tensorflow.python.ops import control_flow_ops

class BaseDataSet(object):

    def __init__(self, flags):
        self.flags = flags


class EstimatorDataSet(BaseDataSet):

    def __init__(self, flags):
        super(EstimatorDataSet, self).__init__(flags)

    def input_fn(self, mode, num_epochs=1):
        raise NotImplementedError

    def parse_record(self, mode, record):
        raise NotImplementedError


class DataSet(EstimatorDataSet):

    def __init__(self, flags):
        super(DataSet, self).__init__(flags)
        self.batch_size = flags.batch_size
        self.multi_gpu = flags.multi_gpu
        self.num_parallel_calls = flags.num_parallel_calls

    def process(self, dataset, mode, shuffle_buffer, num_epochs, examples_per_epoch=None):
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
            if not examples_per_epoch:
                raise ValueError('Must set examples_per_epoch in multi gpu mode')
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


class DetectionDataSet(EstimatorDataSet):

    def __init__(self, flags):
        super(DetectionDataSet, self).__init__(flags)
        self.batch_size = flags.batch_size
        self.num_parallel_calls = flags.num_parallel_calls

    def read_dataset(self, file_read_func, decode_func, input_files, num_epochs=None):
        filenames = tf.concat([tf.matching_files(pattern) for pattern in input_files],
                            0)
        # Shard, shuffle, and read files.
        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
        filename_dataset = filename_dataset.shuffle(self.flags.filenames_shuffle_buffer_size)

        filename_dataset = filename_dataset.repeat(num_epochs)

        records_dataset = filename_dataset.apply(
                tf.contrib.data.parallel_interleave(
                    file_read_func, cycle_length=self.flags.num_readers,
                    block_length=self.flags.read_block_length, sloppy=True))
        records_dataset.shuffle(self.flags.shuffle_buffer_size)
        tensor_dataset = records_dataset.map(decode_func,
                            num_parallel_calls=self.num_parallel_calls)
        return tensor_dataset.prefetch(self.flags.prefetch_size)

    def apply_with_random_selector_tuples(self, x, func, num_cases):
        """Computes func(x, sel), with sel sampled from [0...num_cases-1].

        If both preprocess_vars_cache AND key are the same between two calls, sel will
        be the same value in both calls.

        Args:
            x: A tuple of input tensors.
            func: Python function to apply.
            num_cases: Python int32, number of cases to sample sel from.
        Returns:
            The result of func(x, sel), where func receives the value of the
            selector as a python integer, but sel is sampled dynamically.
        """
        num_inputs = len(x)
        rand_sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)

        # Pass the real x only to one of the func calls.
        tuples = [list() for t in x]
        for case in range(num_cases):
            new_x = [control_flow_ops.switch(t, tf.equal(rand_sel, case))[1] for t in x]
            output = func(tuple(new_x), case)
            for j in range(num_inputs):
                tuples[j].append(output[j])

        for i in range(num_inputs):
            tuples[i] = control_flow_ops.merge(tuples[i])[0]
        return tuple(tuples)

    def apply_with_random_selector(self, x, func, num_cases):
        """Computes func(x, sel), with sel sampled from [0...num_cases-1].

        Args:
            x: input Tensor.
            func: Python function to apply.
            num_cases: Python int32, number of cases to sample sel from.

        Returns:
            The result of func(x, sel), where func receives the value of the
            selector as a python integer, but sel is sampled dynamically.
        """
        sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
        # Pass the real x only to one of the func calls.
        return control_flow_ops.merge([
                func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
                for case in range(num_cases)])[0]

