#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order


def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
  """Returns the subset of variables available in the checkpoint.

  Inspects given checkpoint and returns the subset of variables that are
  available in it.

  TODO(rathodv): force input and output to be a dictionary.

  Args:
    variables: a list or dictionary of variables to find in checkpoint.
    checkpoint_path: path to the checkpoint to restore variables from.
    include_global_step: whether to include `global_step` variable, if it
      exists. Default True.

  Returns:
    A list or dictionary of variables.
  Raises:
    ValueError: if `variables` is not a list or dict.
  """
  ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
  ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
  if not include_global_step:
    ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
  vars_in_ckpt = {}
  for variable_name, variable in sorted(variables.items()):
    if variable_name in ckpt_vars_to_shape_map:
      if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
        vars_in_ckpt[variable_name] = variable
      else:
        tf.logging.warn('Variable [%s] is available in checkpoint, but has an '
                        'incompatible shape with model variable.',
                        variable_name)
    else:
      tf.logging.warn('Variable [%s] is not available in checkpoint',
                      variable_name)
  return vars_in_ckpt
