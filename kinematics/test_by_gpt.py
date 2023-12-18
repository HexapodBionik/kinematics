import numpy as np
from kinematics import RoboticArm
from numpy.testing import assert_allclose


def test_fk_edge_cases():
    # Testing the forward kinematics function with edge case inputs
    a = RoboticArm(np.array([0, 0, 0, 1]),
                   np.array([0, 0, 0, 1]),
                   np.array([0, 0, 0, 1]),
                   0,
                   0)

    # With zero lengths, the end effector should be at the base position
    assert_allclose(a.forward(0, 0, 0),
                    np.array([0, 0, 0, 1]),
                    rtol=1e-8, atol=1e-8)
