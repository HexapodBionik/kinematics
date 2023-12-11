import numpy as np
import math
from kinematics import RoboticArm
from numpy.testing import assert_allclose


def test_fk_1():
    a = RoboticArm(np.array([1, 0, 0, 1]),
                   np.array([1, 0, 0, 1]),
                   np.array([1, 0, 0, 1]),
                   0,
                   -math.pi/2)

    assert_allclose(a.forward(0, 0, 0),
                    np.array([2, 0, -1, 1]),
                    rtol=1e-8, atol=1e-8)

    assert_allclose(a.forward(0, math.pi/2, 0),
                    np.array([2, 0, 1, 1]),
                    rtol=1e-8, atol=1e-8)

    assert_allclose(a.forward(0, 0, math.pi/2),
                    np.array([3, 0, 0, 1]),
                    rtol=1e-8, atol=1e-8)

    assert_allclose(a.forward(math.pi/2, 0, 0),
                    np.array([0, 2, -1, 1]),
                    rtol=1e-8, atol=1e-8)

    assert_allclose(a.forward(-math.pi/2, 0, 0),
                    np.array([0, -2, -1, 1]),
                    rtol=1e-8, atol=1e-8)

    assert_allclose(a.forward(0, -math.pi/2, 0),
                    np.array([0, 0, -1, 1]),
                    rtol=1e-8, atol=1e-8)

    assert_allclose(a.forward(math.pi/2, -math.pi/2, 0),
                    np.array([0, 0, -1, 1]),
                    rtol=1e-8, atol=1e-8)


def test_fk_2():
    a = RoboticArm(np.array([1, 0, 0, 1]),
                   np.array([2, 0, 0, 1]),
                   np.array([3, 0, 0, 1]),
                   0,
                   -math.pi/2)

    assert_allclose(a.forward(0, 0, 0),
                    np.array([3, 0, -3, 1]),
                    rtol=1e-8, atol=1e-8)


def test_fk_3():
    a = RoboticArm(np.array([1, 0, 0, 1]),
                   np.array([1, 0, 0, 1]),
                   np.array([1, 0, 0, 1]),
                   0,
                   0)

    assert_allclose(a.forward(0, 0, 0),
                    np.array([3, 0, 0, 1]),
                    rtol=1e-8, atol=1e-8)

    assert_allclose(a.forward(-math.pi/2, 0, 0),
                    np.array([0, -3, 0, 1]),
                    rtol=1e-8, atol=1e-8)

    assert_allclose(a.forward(0, -math.pi/2, 0),
                    np.array([1, 0, -2, 1]),
                    rtol=1e-8, atol=1e-8)


def test_rk_1():
    a = RoboticArm(np.array([1, 0, 0, 1]),
                   np.array([1, 0, 0, 1]),
                   np.array([1, 0, 0, 1]),
                   0,
                   0)

    assert_allclose(a.reverse(np.array([3, 0, 0, 1])),
                    (0, 0, 0),
                    rtol=1e-8, atol=1e-8)

    assert_allclose(a.reverse(np.array([0, -3, 0, 1])),
                    (-math.pi/2, 0, 0),
                    rtol=1e-8, atol=1e-8)

    assert_allclose(a.reverse(np.array([1, 0, -2, 1])),
                    (0, -math.pi/2, 0),
                    rtol=1e-8, atol=1e-8)


def test_rk_2():
    a = RoboticArm(np.array([1, 0, 0, 1]),
                   np.array([1, 0, 0, 1]),
                   np.array([1, 0, 0, 1]),
                   0,
                   -math.pi/2)
    assert_allclose(a.reverse(np.array([0, 0, -1, 1])),
                    (0, -math.pi/2, 0),
                    rtol=1e-8, atol=1e-8)
