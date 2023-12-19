from kinematics.kinematics import KinematicsSolver
import numpy as np


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
        @return: list of 3 tuples each containing coefficients for cubic polynomial
        realizing trajectory for each joint
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
        Solve system of equations and get the coefficients for a cubic polynomial
        """
        A = np.array([[1, 0, 0, 0],
                      [1, end_time, pow(end_time, 2), pow(end_time, 3)],
                      [0, 1, 0, 0],
                      [0, 1, 2 * end_time, 3 * pow(end_time, 2)]]
                     )
        b = np.array([start_angle, end_angle, start_velocity, end_velocity])

        a = np.linalg.solve(A, b)
        return a
