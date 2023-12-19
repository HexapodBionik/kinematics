import numpy as np
import math
from typing import Tuple


def trans(v):
    """
    Create translation matrix by
    vector v.

    @param v: Some vector from SE3
    @return: Translation by v matrix
    """
    return np.array([[1, 0, 0, v[0]],
                     [0, 1, 0, v[1]],
                     [0, 0, 1, v[2]],
                     [0, 0, 0, v[3]]])


def rot_x(alpha):
    """
    Create rotation matrix by
    angle alpha around axis X.

    @param alpha: Some angle
    @return: Rotation matrix
    """
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(alpha), -np.sin(alpha), 0],
                     [0, np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 0, 1]])


def rot_y(alpha):
    """
    Create rotation matrix by
    angle alpha around axis Y.

    @param alpha: Some angle
    @return: Rotation matrix
    """
    return np.array([[np.cos(alpha), 0, -np.sin(alpha), 0],
                     [0, 1, 0, 0],
                     [np.sin(alpha), 0, np.cos(alpha), 0],
                     [0, 0, 0, 1]])


def rot_z(alpha):
    """
    Create rotation matrix by
    angle alpha around axis Z.

    @param alpha: Some angle
    @return: Rotation matrix
    """
    return np.array([[np.cos(alpha), -np.sin(alpha), 0, 0],
                     [np.sin(alpha), np.cos(alpha), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def se3_norm(v):
    """
    Length of vector from special
    euclidean group SE3.

    @param v: Some vector from SE3
    @return: Length of v
    """
    assert v.ndim == 1
    assert len(v) == 4
    assert v[3] == 1

    return np.linalg.norm(
            np.array([v[0], v[1], v[2]]))


class KinematicsSolver:
    """
    Class storing parameters of the
    robotic leg of Hexapod Robot.
    The leg has 3 degrees of freedom.
    """

    def __init__(self, t1: np.ndarray, t2: np.ndarray, t3: np.ndarray,
                 ma2: float, ma3: float):
        """
        @param t1: Translation 1 (between J1 and J2)
        @param t2: Translation 2 (between J2 and J3)
        @param t3: Translation 3 (between J3 and FCP)
        @param ma2: Servo mount angle 2 (in radians)
        @param ma3: Servo mount angle 3 (in radians)
        """

        assert t1.ndim == 1
        assert t2.ndim == 1
        assert t3.ndim == 1
        assert len(t1) == 4
        assert len(t2) == 4
        assert len(t3) == 4

        assert t1[3] == 1
        assert t2[1] == 0
        assert t2[3] == 1
        assert t3[1] == 0
        assert t3[3] == 1

        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.ma2 = ma2
        self.ma3 = ma3

    def forward(self, q1: float, q2: float, q3: float) -> np.ndarray:
        """
        Forward Kinematics

        @param q1: Joint1 angle (in radians)
        @param q2: Joint2 angle (in radians)
        @param q3: Joint3 angle (in radians)
        @return: Position of leg's foot center point
        """

        m = rot_z(q1)
        m = m.dot(trans(self.t1))
        m = m.dot(rot_x(math.pi/2))
        m = m.dot(rot_z(q2+self.ma2))
        m = m.dot(trans(self.t2))
        m = m.dot(rot_z(q3+self.ma3))
        m = m.dot(trans(self.t3))
        m = m.dot(rot_z(math.pi))

        return m.dot(np.array([0, 0, 0, 1]))

    def inverse(self, v: np.ndarray) -> Tuple[float, float, float]:
        """
        Inverse Kinematics

        @param v: Desired position of arm's end | (x, y, z, 1)
        @return: Angles to be set on servos
        """

        assert v.ndim == 1
        assert len(v) == 4
        assert v[3] == 1

        x = v[0]
        y = v[1]
        z = v[2]

        l1 = se3_norm(self.t1)
        l2 = se3_norm(self.t2)
        l3 = se3_norm(self.t3)

        q1 = (math.pi/2)*np.sign(y) if x == 0 \
            else math.atan2(y, x)

        v = rot_z(-q1).dot(v)
        x = v[0] - self.t1[0]
        y = v[1]
        z = v[2] - self.t1[2]

        q2 = self.ma2
        q2 += (math.pi/2)*np.sign(z) if x == 0 \
            else math.atan2(z, x)

        q2 += np.arccos((l2**2 + x**2 + z**2 - l3**2) /
                        (2*l2*np.sqrt(x**2+z**2)))

        q3 = -self.ma3
        q3 += np.arccos((l2**2 + l3**2 - x**2 - z**2) /
                         (2*l2*l3)) - math.pi

        return q1, q2, q3
