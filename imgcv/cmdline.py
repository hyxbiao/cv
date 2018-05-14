#coding=utf-8
#date: 2018-04-11
#author: hyxbiao

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import re

import tensorflow as tf  # pylint: disable=g-bad-import-order


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__(add_help=True)
        self.add_argument(
            "--model_dir",
            help="The location of the model checkpoint files.",
        )
        self.add_argument(
            "--filter", default=".*",
            help="[default: %(default)s] Regex filter vars",
        )
        self.add_argument(
            '--var', action='store_true', default=False,
            help='Show variables'
        )


class Application(object):
    def __init__(self, flags):
        self.flags = flags

    def get_variables_to_train(self, scopes):
        """Returns a list of variables to train.

        Returns:
          A list of variables to train by the optimizer.
        """
        if not scopes:
            return tf.trainable_variables()
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

    def run(self):
        last_checkpoint_filename = tf.train.latest_checkpoint(self.flags.model_dir)
        if last_checkpoint_filename is None:
            last_checkpoint_filename = self.flags.model_dir
        pattern = self.flags.filter
        print('last checkpoint filename: {}, filter: {}'.format(
            last_checkpoint_filename, pattern))

        '''
        variables_to_train = self.get_variables_to_train([pattern])
        for var in variables_to_train:
            print('variable_name: {}, var: {}'.format(var.op.name, var))
        '''

        reader = tf.train.NewCheckpointReader(last_checkpoint_filename)
        vars_to_shape_map = reader.get_variable_to_shape_map()
        for variable_name in vars_to_shape_map:
            var = vars_to_shape_map[variable_name]
            if re.match(pattern, variable_name):
                print('variable_name: {}, var: {}'.format(variable_name, var))


def imgcv_run(argv=None):

    if argv is None:
        argv = sys.argv
    parser = ArgParser()
    flags = parser.parse_args(args=argv[1:])

    app = Application(flags)
    app.run()

