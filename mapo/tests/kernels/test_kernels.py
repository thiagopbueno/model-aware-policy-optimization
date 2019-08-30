# pylint: disable=invalid-name, missing-docstring, redefined-outer-name

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Dense

from mapo.kernels import _flat_and_concat, l1_kernel, l2_kernel, linf_kernel, cos_kernel


@pytest.fixture
def fcnet():
    model = Sequential(
        [
            Dense(32, input_shape=(2,)),
            BatchNormalization(),
            Activation("relu"),
            Dense(64),
            BatchNormalization(),
            Activation("relu"),
            Dense(2),
        ]
    )
    return model


def test_l1_kernel(fcnet):
    u = fcnet.trainable_variables
    v = [2 * x for x in u]

    d1 = l1_kernel(u, v, flat=False)
    d2 = l1_kernel(u, v, flat=True)
    assert d1.shape == tuple()
    assert d2.shape == tuple()

    expected_d1 = sum([tf.norm(x, ord=1) for x in u])
    expected_d2 = tf.norm(_flat_and_concat(u), ord=1)

    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        d1_, expected_d1_ = sess.run([d1, expected_d1])
        d2_, expected_d2_ = sess.run([d2, expected_d2])
        assert np.all(d1_ >= 0.0)
        assert np.all(d2_ >= 0.0)
        assert np.allclose(d1_, expected_d1_)
        assert np.allclose(d1_, expected_d2_)
        assert np.allclose(d2_, expected_d2_)
        assert np.allclose(d2_, expected_d1_)


@pytest.mark.skip
def test_l2_kernel(fcnet):
    u = fcnet.trainable_variables
    v = [2 * x for x in u]

    d1 = l2_kernel(u, v, flat=False)
    d2 = l2_kernel(u, v, flat=True)
    assert d1.shape == tuple()
    assert d2.shape == tuple()

    expected_d1 = sum([tf.norm(x, ord=2) for x in u])
    expected_d2 = tf.norm(_flat_and_concat(u), ord=2)

    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        d1_, expected_d1_ = sess.run([d1, expected_d1])
        d2_, expected_d2_ = sess.run([d2, expected_d2])
        assert np.all(d1_ >= 0.0)
        assert np.all(d2_ >= 0.0)
        assert np.allclose(d1_, expected_d1_)
        assert not np.allclose(d1_, expected_d2_)
        assert np.allclose(d2_, expected_d2_)
        assert not np.allclose(d2_, expected_d1_)


def test_linf_kernel(fcnet):
    u = fcnet.trainable_variables
    v = [2 * x for x in u]

    d1 = linf_kernel(u, v, flat=False)
    d2 = linf_kernel(u, v, flat=True)
    assert d1.shape == tuple()
    assert d2.shape == tuple()

    expected_d1 = sum([tf.norm(x, ord=np.inf) for x in u])
    expected_d2 = tf.norm(_flat_and_concat(u), ord=np.inf)

    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        d1_, expected_d1_ = sess.run([d1, expected_d1])
        d2_, expected_d2_ = sess.run([d2, expected_d2])
        assert np.all(d1_ >= 0.0)
        assert np.all(d2_ >= 0.0)
        assert np.allclose(d1_, expected_d1_)
        assert not np.allclose(d1_, expected_d2_)
        assert np.allclose(d2_, expected_d2_)
        assert not np.allclose(d2_, expected_d1_)


def test_cos_kernel(fcnet):
    # pylint: disable=too-many-locals
    u = fcnet.trainable_variables

    v1 = [np.random.uniform(low=0.0, high=1.0) * x for x in u]
    v2 = [np.random.uniform(low=1.0, high=10000) * x for x in u]
    v3 = [np.random.uniform(low=-10000, high=-1.0) * x for x in u]
    v4 = [np.random.uniform(low=-1.0, high=0.0) * x for x in u]

    sim0 = cos_kernel(u, u, flat=False)
    sim1 = cos_kernel(u, v1, flat=False)
    sim2 = cos_kernel(u, v2, flat=False)
    sim3 = cos_kernel(u, v3, flat=False)
    sim4 = cos_kernel(u, v4, flat=False)

    assert sim1.shape == tuple()
    assert sim2.shape == tuple()
    assert sim3.shape == tuple()
    assert sim4.shape == tuple()

    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())

        sim0_ = sess.run(sim0)
        # bias and batch beta initialized with zeros
        assert np.allclose(sim0_, -len(u) / 2)

        sim1_, sim2_ = sess.run([sim1, sim2])
        assert np.allclose(sim1_, sim2_)

        sim3_, sim4_ = sess.run([sim3, sim4])
        assert np.allclose(sim3_, sim4_)
