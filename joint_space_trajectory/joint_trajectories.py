from kinematics.kinematics import KinematicsSolver
import numpy as np
import math
from typing import Tuple


class SimpleJointSpaceTrajectory:
    """
    Class used for generation of simple trajectory of hexapod leg in joint space.
    Trajectory generation using 3rd order polynomial.
    """

    def __init__(self, kinematics_solver: KinematicsSolver):
        self._kinematics_solver = kinematics_solver

    def generate_trajectory(self, start_coordinates: np.ndarray,
                            end_coordinates: np.ndarray,
                            velocity_start: np.ndarray,
                            velocity_end: np.ndarray, time_end: float) -> list:
        """
        Generate coefficients for cubic polynomial realizing leg's trajectory
        @param start_coordinates: start coordinates of leg's trajectory
        provided as tuple (x, y, z)
        @param end_coordinates: end coordinates of leg's trajectory
        provided as tuple (x, y, z)
        @param velocity_start: start velocity of each joint in (rad/s)
        @param velocity_end: end velocity of each joint in (rad/s)
        @param time_end: time of movement from start to end
        coordinates in seconds
        @return: list of 3 tuples each containing coefficients
        for cubic polynomial realizing trajectory for each joint
        """
        start_position = np.array([*start_coordinates, 1])
        end_position = np.array([*end_coordinates, 1])

        start_angles_array = self._kinematics_solver.inverse(start_position)
        end_angles_array = self._kinematics_solver.inverse(end_position)

        poly_coefficients = []
        for i in range(3):
            poly_coefficients.append(
                self._generate_polynomial_coefficients(start_angles_array[i],
                                                       end_angles_array[i],
                                                       float(velocity_start[i]),
                                                       float(velocity_end[i]),
                                                       time_end))

        return poly_coefficients

    def _generate_polynomial_coefficients(self, start_angle: float,
                                          end_angle: float,
                                          start_velocity: float,
                                          end_velocity: float, end_time: float) -> np.ndarray:
        """
        Solve system of equations and get the coefficients
        for a cubic polynomial
        """
        A = np.array([[1, 0, 0, 0],
                      [1, end_time, pow(end_time, 2), pow(end_time, 3)],
                      [0, 1, 0, 0],
                      [0, 1, 2 * end_time, 3 * pow(end_time, 2)]]
                     )
        b = np.array([start_angle, end_angle, start_velocity, end_velocity])

        a = np.linalg.solve(A, b)
        return a


class SplinePolynomial:
    def __init__(self, coefficients_list: np.ndarray, time_limits: np.ndarray):
        self._coefficients_list = coefficients_list
        self._time_limits = time_limits
        self._calculated_time_limits = [sum(self._time_limits[:i]) for i in range(1, len(self._time_limits)+1)]

    def __call__(self, time_steps: np.ndarray) -> float:
        values = []
        for time_step in time_steps:
            poly, polynumber = self._select_polynomial(time_step)

            values.append(
                self._generate_polynomial_value(poly, time_step, polynumber))

        return np.array(values)

    def _select_polynomial(self, time_step: float) -> Tuple[np.poly1d, int] | None:
        for i, time_limit in enumerate(self._calculated_time_limits):
            if time_step < time_limit:
                polynumber = i

                poly = np.poly1d(np.flip(self._coefficients_list[polynumber]))
                return poly, polynumber
        return None
    
    def _generate_polynomial_value(self, poly: np.poly1d, time_step: float, polynumber: int) -> float:
        if polynumber == 0:
            return poly(time_step)
        else:
            return poly(
                time_step - self._calculated_time_limits[polynumber - 1])

    def get_velocity(self, time_steps: np.ndarray) -> float:
        values = []
        for time_step in time_steps:
            poly, polynumber = self._select_polynomial(time_step)
            poly = np.polyder(poly)

            values.append(
                self._generate_polynomial_value(poly, time_step, polynumber))
        return np.array(values)

    def get_acceleration(self, time_steps: np.ndarray) -> float:
        values = []
        for time_step in time_steps:
            poly, polynumber = self._select_polynomial(time_step)
            poly = np.polyder(poly, 2)

            values.append(
                self._generate_polynomial_value(poly, time_step, polynumber))
        return np.array(values)


class ViaPointsJointSpaceTrajectory:
    def __init__(self, kinematics_solver: KinematicsSolver):
        self._kinematics_solver = kinematics_solver
        self._poly_deg = 3

    def generate_trajectory(self, coordinates: np.ndarray, velocities: np.ndarray, time_steps: np.ndarray) -> list:
        thetas1 = []
        thetas2 = []
        thetas3 = []
        for coordinate in coordinates:
            start_position = np.array([*coordinate, 1])
            angles = self._kinematics_solver.inverse(start_position)
            thetas1.append(angles[0])
            thetas2.append(angles[1])
            thetas3.append(angles[2])

        joint_angles = [thetas1, thetas2, thetas3]

        polys = []
        for i in range(3):
            coefficients = self._generate_polynomial_coefficients(joint_angles[i], velocities[i], time_steps)
            coeffs_split = np.array_split(coefficients, len(coefficients) // (self._poly_deg+1))
            coeffs_split = [coeffs.reshape((4,)) for coeffs in coeffs_split]
            polys.append(SplinePolynomial(coeffs_split, time_steps))

        return polys

    def _generate_polynomial_coefficients(self, angles: np.ndarray, velocities: np.ndarray,  time_steps: np.ndarray) -> np.ndarray:
        polys = len(time_steps)
        # A = np.zeros((polys*(self._poly_deg+1)))
        A = np.zeros((polys*(self._poly_deg+1), polys*(self._poly_deg+1)))

        # Fill the A matrix with coefficients from position equations
        for i in range(polys):
            A[i*2, i*4] = 1
            for j in range(self._poly_deg+1):
                A[i * 2 + 1, i * 4 + j] = pow(time_steps[i], j)

        A[2*polys, 1] = 1
        for j in range(self._poly_deg):
            A[2*polys + 1, (polys - 1) * 4 + 1 + j] = (1+j)*pow(time_steps[-1], j)

        for i in range(polys-1):
            for j in range(self._poly_deg):
                A[polys*2+2+i, i*4+1 + j] = (1+j)*pow(time_steps[i], j)
            A[polys*2+2+i, 1 + (1+i)*4] = -1

        for i in range(polys-1):
            for j in range(self._poly_deg-1):
                A[polys*3+1+i, i*4+2 + j] = math.factorial(2+j) * pow(time_steps[i], j)
            A[polys*3+1+i, 2 + (1+i)*4] = -2

        b = np.zeros((polys*4, 1))
        b[0] = angles[0]
        for i in range(len(angles)-2):
            b[1+i*2] = angles[1+i]
            b[2+i*2] = angles[1+i]

        b[1+(len(angles)-2)*2] = angles[-1]
        b[2 + (len(angles) - 2) * 2] = velocities[0]
        b[3 + (len(angles) - 2) * 2] = velocities[1]

        a = np.linalg.solve(A, b)
        return a
