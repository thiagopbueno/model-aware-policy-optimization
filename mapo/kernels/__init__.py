"""Collection of kernels for comparing gradient vectors."""
# pylint: disable=invalid-name
import numpy as np
import tensorflow as tf


def _flat_and_concat(u):
    return tf.concat([tf.reshape(x, shape=(-1,)) for x in u], axis=0)


def _lp_kernel(u, v, flat, ord):  # pylint: disable=redefined-builtin
    if flat:
        u = _flat_and_concat(u)
        v = _flat_and_concat(v)
        kernel = tf.norm(u - v, ord=ord)
    else:
        kernel = sum(tf.norm(u_i - v_i, ord=ord) for u_i, v_i in zip(u, v))

    return kernel


def l1_kernel(u, v, flat=False):
    """Returns l-1 distance between `u` and `v` tensors."""
    return _lp_kernel(u, v, flat, 1)


def l2_kernel(u, v, flat=False):
    """Returns l-2 distance between `u` and `v` tensors."""
    if flat:
        u = _flat_and_concat(u)
        v = _flat_and_concat(v)
        kernel = tf.reduce_sum((u - v) ** 2)
    else:
        kernel = sum(tf.reduce_sum((u_i - v_i) ** 2) for (u_i, v_i) in zip(u, v))
    return kernel
    # return _lp_kernel(u, v, flat, 2)


def linf_kernel(u, v, flat=False):
    """Returns l-infinity distance between `u` and `v` tensors."""
    return _lp_kernel(u, v, flat, np.inf)


def _similarity_kernel(u, v, flat=False, unit_vectors=False):
    if flat:
        u = _flat_and_concat(u)
        v = _flat_and_concat(v)

        if unit_vectors:
            u = tf.math.l2_normalize(u)
            v = tf.math.l2_normalize(v)

        kernel = -tf.reduce_sum(u * v)
    else:

        if unit_vectors:
            u = [tf.math.l2_normalize(u_i) for u_i in u]
            v = [tf.math.l2_normalize(v_i) for v_i in v]

        kernel = sum(-tf.reduce_sum(u_i * v_i) for u_i, v_i in zip(u, v))

    return kernel


def dot_product_kernel(u, v, flat=False):
    """Returns dot-product similarity between `u` and `v` tensors."""
    return _similarity_kernel(u, v, flat, unit_vectors=False)


def cos_kernel(u, v, flat=False):
    """Returns cosine similarity between `u` and `v` tensors."""
    return _similarity_kernel(u, v, flat, unit_vectors=True)


KERNELS = {
    "l1": l1_kernel,
    "l2": l2_kernel,
    "linf": linf_kernel,
    "dot-product": dot_product_kernel,
    "cos-similarity": cos_kernel,
}
