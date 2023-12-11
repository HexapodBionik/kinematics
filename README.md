# Kinematics

## Overview

The Kinematics project is a Python implementation of forward and reverse kinematics for robotic arms in "Hexapod" robot. The module includes essential functions for transforming matrices for translation and rotation in three-dimensional space, normalizing vectors in the special Euclidean group SE3, and a `RoboticArm` class modeling a simple three-segment robotic arm with servo joints.

The codebase is succinct and prioritizes efficiency and precision through the use of the NumPy library for matrix operations and the math module for mathematical computations.

## Files in the Project

- `kinematics.py`: Central module providing kinematics functionality.

- `test_kinematics.py`: Unit tests module for verifying the correctness of the kinematic algorithms.

## Installation

This project requires Python 3 and the `numpy` library. You can install the dependencies by running:

```bash
pip install numpy
```

## Usage

To make use of the kinematics module, you can import it in your existing project or run it standalone for custom computations.

```python
from kinematics.kinematics import RoboticArm
```

### Main Features
- **Forward Kinematics:** The `RoboticArm` class allows the user to simulate a three-segment robotic arm with the method `forward(q1, q2, q3)`
- **Reverse Kinematics:** The `RoboticArm` class allows the user to compute angles in three-segment robotic arm in certain position. It can be done with the method `reverse(v)`

## Running Tests

To run unit tests, navigate to the root directory of the project and execute:

```bash
pytest
```

Please ensure that your Python environment has the required packages installed.
