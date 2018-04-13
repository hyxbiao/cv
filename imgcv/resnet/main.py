# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.runner import EstimatorRunner
from imgcv.resnet.model import Model
from imgcv.utils.arg_parsers import parsers
from imgcv.utils.logs import hooks_helper

class ArgParser(argparse.ArgumentParser):
    def __init__(self, resnet_size_choices=None):
        super(ArgParser, self).__init__(parents=[
            parsers.BaseParser(),
            parsers.PerformanceParser(),
            parsers.ImageModelParser(),
            parsers.ExportParser(),
            parsers.BenchmarkParser(),
            parsers.DebugParser(),
        ])

        self.add_argument(
            '--version', '-v', type=int, choices=[1, 2],
            default=Model.DEFAULT_VERSION,
            help='Version of ResNet. (1 or 2) See README.md for details.'
        )

        self.add_argument(
            '--resnet_size', '-rs', type=int, default=50,
            choices=resnet_size_choices,
            help='[default: %(default)s] The size of the ResNet model to use.',
            metavar='<RS>' if resnet_size_choices is None else None
        )


class Runner(EstimatorRunner):

    def __init__(self, flags, estimator, dataset, shape=None):
        super(Runner, self).__init__(flags, estimator, dataset)

        self.model_function = estimator.model_fn
        self.input_function = dataset.input_fn
        self.shape = shape
        self.classifier = self.setup()

    def setup(self):
        # Using the Winograd non-fused algorithms provides a small performance boost.
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        if self.flags.multi_gpu:
            self.validate_batch_size_for_multi_gpu(self.flags.batch_size)

        # There are two steps required if using multi-GPU: (1) wrap the model_fn,
        # and (2) wrap the optimizer. The first happens here, and (2) happens
        # in the model_fn itself when the optimizer is defined.
        self.model_function = tf.contrib.estimator.replicate_model_fn(
            self.model_function,
            loss_reduction=tf.losses.Reduction.MEAN)

        # Create session config based on values of inter_op_parallelism_threads and
        # intra_op_parallelism_threads. Note that we default to having
        # allow_soft_placement = True, which is required for multi-GPU and not
        # harmful for other modes.
        session_config = tf.ConfigProto(
            inter_op_parallelism_threads=self.flags.inter_op_parallelism_threads,
            intra_op_parallelism_threads=self.flags.intra_op_parallelism_threads,
            allow_soft_placement=True)

        # Set up a RunConfig to save checkpoint and set session config.
        run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                                    session_config=session_config)
        classifier = tf.estimator.Estimator(
            model_fn=self.model_function, model_dir=self.flags.model_dir, config=run_config,
            params={
              'resnet_size': self.flags.resnet_size,
              'data_format': self.flags.data_format,
              'batch_size': self.flags.batch_size,
              'multi_gpu': self.flags.multi_gpu,
              'version': self.flags.version,
            })
        return classifier

    def run_internal(self):
        if self.flags.benchmark_log_dir is not None:
            benchmark_logger = logger.BenchmarkLogger(self.flags.benchmark_log_dir)
            benchmark_logger.log_run_info("resnet")
        else:
            benchmark_logger = None

        for _ in range(self.flags.train_epochs // self.flags.epochs_between_evals):
            train_hooks = hooks_helper.get_train_hooks(
                self.flags.hooks,
                batch_size=self.flags.batch_size,
                benchmark_log_dir=self.flags.benchmark_log_dir)

            print('Starting a training cycle.')

            def input_fn_train():
                return self.input_function(tf.estimator.ModeKeys.TRAIN, self.flags.epochs_between_evals)

            self.classifier.train(input_fn=input_fn_train, hooks=train_hooks,
                             max_steps=self.flags.max_train_steps)

            print('Starting to evaluate.')
            # Evaluate the model and print results
            def input_fn_eval():
                return self.input_function(tf.estimator.ModeKeys.EVAL, 1)

            # self.flags.max_train_steps is generally associated with testing and profiling.
            # As a result it is frequently called with synthetic data, which will
            # iterate forever. Passing steps=self.flags.max_train_steps allows the eval
            # (which is generally unimportant in those circumstances) to terminate.
            # Note that eval will run for max_train_steps each loop, regardless of the
            # global_step count.
            eval_results = self.classifier.evaluate(input_fn=input_fn_eval,
                                               steps=self.flags.max_train_steps)
            print(eval_results)

            if benchmark_logger:
                benchmark_logger.log_estimator_evaluation_result(eval_results)

        if self.flags.export_dir is not None:
            self.warn_on_multi_gpu_export(self.flags.multi_gpu)

            # Exports a saved model for the given classifier.
            input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
            self.shape, batch_size=self.flags.batch_size)
            self.classifier.export_savedmodel(self.flags.export_dir, input_receiver_fn)

    def warn_on_multi_gpu_export(self, multi_gpu=False):
        """For the time being, multi-GPU mode does not play nicely with exporting."""
        if multi_gpu:
            tf.logging.warning(
                'You are exporting a SavedModel while in multi-GPU mode. Note that '
                'the resulting SavedModel will require the same GPUs be available.'
                'If you wish to serve the SavedModel from a different device, '
                'try exporting the SavedModel with multi-GPU mode turned off.')

    def validate_batch_size_for_multi_gpu(self, batch_size):
        """For multi-gpu, batch-size must be a multiple of the number of GPUs.

        Note that this should eventually be handled by replicate_model_fn
        directly. Multi-GPU support is currently experimental, however,
        so doing the work here until that feature is in place.

        Args:
        batch_size: the number of examples processed in each training batch.

        Raises:
        ValueError: if no GPUs are found, or selected batch_size is invalid.
        """
        from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top

        local_device_protos = device_lib.list_local_devices()
        num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
        if not num_gpus:
            raise ValueError('Multi-GPU mode was specified, but no GPUs '
                         'were found. To use CPU, run without --multi_gpu.')

        remainder = batch_size % num_gpus
        if remainder:
            err = ('When running with multiple GPUs, batch size '
               'must be a multiple of the number of available GPUs. '
               'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
              ).format(num_gpus, batch_size, batch_size - remainder)
            raise ValueError(err)


