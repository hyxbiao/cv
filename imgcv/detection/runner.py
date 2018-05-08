from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf  # pylint: disable=g-bad-import-order

from imgcv.runner import EstimatorRunner
from imgcv.utils.arg_parsers import parsers
from imgcv.utils.export import export

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__(parents=[
            parsers.BaseParser(),
            parsers.GPUParser(),
            parsers.PerformanceParser(),
            parsers.ImageModelParser(),
            parsers.ExportParser(),
            parsers.BenchmarkParser(),
            parsers.PredictParser(),
            parsers.PreTrainParser(),
        ])


class Runner(EstimatorRunner):

    def __init__(self, flags, estimator, dataset, shape=None):
        super(Runner, self).__init__(flags, estimator, dataset)

        self.model_function = estimator.model_fn
        self.input_function = dataset.input_fn
        self.shape = shape

    def run(self):
        self.setup()
        if self.flags.export_dir is not None:
            self.export()
            return
        if self.flags.predict:
            return self.predict()
        self.train()

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

        if self.flags.gpu_allow_growth:
            session_config.gpu_options.allow_growth = True
        else:
            session_config.gpu_options.per_process_gpu_memory_fraction = self.flags.gpu_memory_fraction

        # Set up a RunConfig to save checkpoint and set session config.
        run_config = tf.estimator.RunConfig().replace(
                #save_checkpoints_secs=3600,
                save_summary_steps=3,
                save_checkpoints_steps=30,
                log_step_count_steps=10,
                session_config=session_config,
                keep_checkpoint_max=3)
        ws = None
        if self.flags.pretrain_model_dir:
            ws = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=self.flags.pretrain_model_dir,
                vars_to_warm_start=self.flags.pretrain_warm_vars)
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_function, model_dir=self.flags.model_dir, config=run_config,
            params={
            },
            warm_start_from=ws)

        if self.flags.benchmark_log_dir is not None:
            self.benchmark_logger = logger.BenchmarkLogger(self.flags.benchmark_log_dir)
            self.benchmark_logger.log_run_info("runner")
        else:
            self.benchmark_logger = None

    def train(self, do_eval=False):
        for _ in range(self.flags.train_epochs // self.flags.epochs_between_evals):
            tf.logging.info('Starting a training cycle.')

            def input_fn_train():
                return self.input_function(tf.estimator.ModeKeys.TRAIN, self.flags.epochs_between_evals)

            self.estimator.train(input_fn=input_fn_train,
                             max_steps=self.flags.max_train_steps)

            if not do_eval:
                continue
            tf.logging.info('Starting to evaluate.')
            # Evaluate the model and print results
            def input_fn_eval():
                return self.input_function(tf.estimator.ModeKeys.EVAL, 1)

            # self.flags.max_train_steps is generally associated with testing and profiling.
            # As a result it is frequently called with synthetic data, which will
            # iterate forever. Passing steps=self.flags.max_train_steps allows the eval
            # (which is generally unimportant in those circumstances) to terminate.
            # Note that eval will run for max_train_steps each loop, regardless of the
            # global_step count.
            eval_results = self.estimator.evaluate(input_fn=input_fn_eval,
                                               steps=self.flags.max_train_steps)
            tf.logging.info(eval_results)

            if self.benchmark_logger:
                self.benchmark_logger.log_estimator_evaluation_result(eval_results)

    def predict(self):
        def input_fn_predict():
            return self.input_function(tf.estimator.ModeKeys.PREDICT)
        predict_outputs = self.estimator.predict(input_fn=input_fn_predict,
                yield_single_examples=self.flags.predict_yield_single)
        return predict_outputs

    def export(self):
        self.warn_on_multi_gpu_export(self.flags.multi_gpu)

        # Exports a saved model for the given estimator.
        input_receiver_fn = export.build_serving_input_receiver_fn(
            self.shape, batch_size=self.flags.batch_size)
        self.estimator.export_savedmodel(self.flags.export_dir, input_receiver_fn)

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


