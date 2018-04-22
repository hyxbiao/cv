
from __future__ import print_function

import tensorflow as tf
import numpy as np

debug = False

#https://github.com/balancap/SSD-Tensorflow/blob/master/tf_extended/metrics.py
def average_precision_voc12(precision, recall, name=None):
    """Compute (interpolated) average precision from precision and recall Tensors.
    The implementation follows Pascal 2012 and ILSVRC guidelines.
    See also: https://sanchom.wordpress.com/tag/average-precision/
    """
    with tf.name_scope(name, 'average_precision_voc12', [precision, recall]):
        # Convert to float64 to decrease error on Riemann sums.
        precision = tf.cast(precision, dtype=tf.float64)
        recall = tf.cast(recall, dtype=tf.float64)

        # Add bounds values to precision and recall.
        precision = tf.concat([[0.], precision, [0.]], axis=0)
        recall = tf.concat([[0.], recall, [1.]], axis=0)
        # Ensures precision is increasing in reverse order.
        precision = tfe_math.cummax(precision, reverse=True)

        # Riemann sums for estimating the integral.
        # mean_pre = (precision[1:] + precision[:-1]) / 2.
        mean_pre = precision[1:]
        diff_rec = recall[1:] - recall[:-1]
        ap = tf.reduce_sum(mean_pre * diff_rec)
        return ap

def tensor(y_true, y_pred):
    #y_true = np.array([[2], [1], [0], [3], [0], [1]]).astype(np.int64)
    y_true = tf.identity(y_true)

    '''
    y_pred = np.array([[0.1, 0.2, 0.6, 0.1],
                       [0.8, 0.05, 0.1, 0.05],
                       [0.3, 0.4, 0.1, 0.2],
                       [0.6, 0.25, 0.1, 0.05],
                       [0.1, 0.2, 0.6, 0.1],
                       [0.9, 0.0, 0.03, 0.07]]).astype(np.float32)
    '''
    y_pred = tf.identity(y_pred)

    _, m_ap = tf.metrics.average_precision_at_k(y_true, y_pred, 1)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())

    print(sess.run(y_true))
    print(sess.run(y_pred))

    tf_map = sess.run(m_ap)
    print(tf_map)

def raw(y_true, y_pred):
    _, classes = y_true.shape

    average_precisions = []

    for index in range(classes):
        row_indices_sorted = np.argsort(-y_pred[:, index])

        y_true_cls = y_true[row_indices_sorted, index]
        y_pred_cls = y_pred[row_indices_sorted, index]

        if debug:
            print('y_true_cls:', y_true_cls)
            print('y_pred_cls:', y_pred_cls)

        tp = (y_true_cls == 1)
        fp = (y_true_cls == 0)

        if debug:
            print('tp:', tp)
            print('fp:', fp)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        if debug:
            print('cumsum tp:', tp)
            print('cumsum fp:', fp)

        npos = np.sum(y_true_cls)

        if debug:
            print('npos:', npos)
        rec = tp*1.0 / npos

        if debug:
            print('rec:', rec)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp*1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)
        if debug:
            print('prec:', prec)

        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        if debug:
            print('mrec:', mrec)
            print('mpre:', mpre)

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        if debug:
            print('update mpre:', mpre)

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        if debug:
            print('change i:', i, i+1, mrec[i+1], mrec[i])

        # and sum (\Delta recall) * prec
        average_precisions.append(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

    print(average_precisions)


def main():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])

    y_true = np.expand_dims(y_true, axis=1)
    y_pred = np.expand_dims(y_pred, axis=1)

    raw(y_true, y_pred)
    tensor(y_true, y_pred)

main()
