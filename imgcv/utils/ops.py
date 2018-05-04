#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf  # pylint: disable=g-bad-import-order

def padded_one_hot_encoding(indices, depth, left_pad):
  """Returns a zero padded one-hot tensor.

  This function converts a sparse representation of indices (e.g., [4]) to a
  zero padded one-hot representation (e.g., [0, 0, 0, 0, 1] with depth = 4 and
  left_pad = 1). If `indices` is empty, the result will simply be a tensor of
  shape (0, depth + left_pad). If depth = 0, then this function just returns
  `None`.

  Args:
    indices: an integer tensor of shape [num_indices].
    depth: depth for the one-hot tensor (integer).
    left_pad: number of zeros to left pad the one-hot tensor with (integer).

  Returns:
    padded_onehot: a tensor with shape (num_indices, depth + left_pad). Returns
      `None` if the depth is zero.

  Raises:
    ValueError: if `indices` does not have rank 1 or if `left_pad` or `depth are
      either negative or non-integers.

  TODO(rathodv): add runtime checks for depth and indices.
  """
  if depth < 0 or not isinstance(depth, six.integer_types):
    raise ValueError('`depth` must be a non-negative integer.')
  if left_pad < 0 or not isinstance(left_pad, six.integer_types):
    raise ValueError('`left_pad` must be a non-negative integer.')
  if depth == 0:
    return None

  rank = len(indices.get_shape().as_list())
  if rank != 1:
    raise ValueError('`indices` must have rank 1, but has rank=%s' % rank)

  def one_hot_and_pad():
    one_hot = tf.cast(tf.one_hot(tf.cast(indices, tf.int64), depth,
                                 on_value=1, off_value=0), tf.float32)
    return tf.pad(one_hot, [[0, 0], [left_pad, 0]], mode='CONSTANT')
  result = tf.cond(tf.greater(tf.size(indices), 0), one_hot_and_pad,
                   lambda: tf.zeros((depth + left_pad, 0)))
  return tf.reshape(result, [-1, depth + left_pad])

def meshgrid(x, y):
  """Tiles the contents of x and y into a pair of grids.

  Multidimensional analog of numpy.meshgrid, giving the same behavior if x and y
  are vectors. Generally, this will give:

  xgrid(i1, ..., i_m, j_1, ..., j_n) = x(j_1, ..., j_n)
  ygrid(i1, ..., i_m, j_1, ..., j_n) = y(i_1, ..., i_m)

  Keep in mind that the order of the arguments and outputs is reverse relative
  to the order of the indices they go into, done for compatibility with numpy.
  The output tensors have the same shapes.  Specifically:

  xgrid.get_shape() = y.get_shape().concatenate(x.get_shape())
  ygrid.get_shape() = y.get_shape().concatenate(x.get_shape())

  Args:
    x: A tensor of arbitrary shape and rank. xgrid will contain these values
       varying in its last dimensions.
    y: A tensor of arbitrary shape and rank. ygrid will contain these values
       varying in its first dimensions.
  Returns:
    A tuple of tensors (xgrid, ygrid).
  """
  with tf.name_scope('Meshgrid'):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
    y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

    xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
    new_shape = y.get_shape().concatenate(x.get_shape())
    xgrid.set_shape(new_shape)
    ygrid.set_shape(new_shape)

    return xgrid, ygrid

def expanded_shape(orig_shape, start_dim, num_dims):
  """Inserts multiple ones into a shape vector.

  Inserts an all-1 vector of length num_dims at position start_dim into a shape.
  Can be combined with tf.reshape to generalize tf.expand_dims.

  Args:
    orig_shape: the shape into which the all-1 vector is added (int32 vector)
    start_dim: insertion position (int scalar)
    num_dims: length of the inserted all-1 vector (int scalar)
  Returns:
    An int32 vector of length tf.size(orig_shape) + num_dims.
  """
  with tf.name_scope('ExpandedShape'):
    start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
    before = tf.slice(orig_shape, [0], start_dim)
    add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
    after = tf.slice(orig_shape, start_dim, [-1])
    new_shape = tf.concat([before, add_shape, after], 0)
    return new_shape


