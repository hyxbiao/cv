#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf  # pylint: disable=g-bad-import-order

def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def assert_shape_equal(shape_a, shape_b):
  """Asserts that shape_a and shape_b are equal.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
  mismatch.

  Args:
    shape_a: a list containing shape of the first tensor.
    shape_b: a list containing shape of the second tensor.

  Returns:
    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
    when the shapes are dynamic.

  Raises:
    ValueError: When shapes are both static and unequal.
  """
  if (all(isinstance(dim, int) for dim in shape_a) and
      all(isinstance(dim, int) for dim in shape_b)):
    if shape_a != shape_b:
      raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
    else: return tf.no_op()
  else:
    return tf.assert_equal(shape_a, shape_b)

