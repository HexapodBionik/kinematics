import numpy as np
from typing import Tuple


def acquire_trajectory_functions(poly_coefficients: np.ndarray) \
        -> Tuple[np.poly1d, np.poly1d, np.poly1d]:
    position_polynomial = np.poly1d(np.flip(poly_coefficients))
    velocity_polynomial = np.polyder(position_polynomial)
    acceleration_polynomial = np.polyder(velocity_polynomial)
    return position_polynomial, velocity_polynomial, acceleration_polynomial
