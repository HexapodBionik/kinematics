from kinematics.kinematics import RoboticArm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

# Wszystkie wartości translacji w metrach
# Translacja T1 - pomiędzy J1, a J2
t1 = np.array([0.05, -0.035, 0, 1])

# Translacja T2 - pomiędzy J2, a J3
t2 = np.array([0.09, 0, 0, 1])

# Translacja T3 - pomiędzy J3, a FCP
t3 = np.array([0.1, 0, 0, 1])


a = RoboticArm(t1, t2, t3, 0, -np.pi/2)


class JointSpaceTrajectory:
    def __init__(self, kinematics_solver: RoboticArm):
        self._kinematics_solver = kinematics_solver

    def generate_trajectory(self, start_coordinates: np.ndarray,
                            end_coordinates: np.ndarray,
                            velocity_start: np.ndarray,
                            velocity_end: np.ndarray, time_end: float):
        start_position = start_coordinates
        end_position = end_coordinates
        theta0_array = self._kinematics_solver.reverse(start_position)
        thetaf_array = self._kinematics_solver.reverse(end_position)

        print([x * 180 / np.pi for x in theta0_array])
        print([x * 180 / np.pi for x in thetaf_array])

        coefficients = []
        for i in range(3):
            coefficients.append(
                self.generate_polynomial_coefficients(theta0_array[i],
                                                      thetaf_array[i],
                                                      velocity_start[i],
                                                      velocity_end[i],
                                                      time_end))

        return coefficients

    def generate_polynomial_coefficients(self, theta_start, theta_end,
                                         velocity_start, velocity_end,
                                         time_end):
        A = np.array([[1, 0, 0, 0],
                      [1, time_end, pow(time_end, 2), pow(time_end, 3)],
                      [0, 1, 0, 0],
                      [0, 1, 2 * time_end, 3 * pow(time_end, 2)]]
                     )
        b = np.array([theta_start, theta_end, velocity_start, velocity_end])

        a = np.linalg.solve(A, b)
        return a


trajectory = JointSpaceTrajectory(a)
start_coordinates = np.array([-0.02, 0, 0, 1])
end_coordinates = np.array([0.02, 0, 0, 1])

velocity_start = np.array([0, 0, 0])
velocity_end = np.array([0, 0, 0])
time_end = 3


coefficients = trajectory.generate_trajectory(start_coordinates, end_coordinates, velocity_start, velocity_end, time_end)

x = np.arange(0, time_end, 0.05)

j_pos = []
j_vel = []
j_accel = []

for coeffs in coefficients:
    pos_poly = np.poly1d(np.flip(coeffs))
    vel_poly = np.polyder(pos_poly)
    accel_poly = np.polyder(vel_poly)

    j_pos.append(pos_poly)
    j_vel.append(vel_poly)
    j_accel.append(accel_poly)


x_points = []
y_points = []
z_points = []

theta1 = j_pos[0](x)
theta2 = j_pos[1](x)
theta3 = j_pos[2](x)
for i in range(len(x)):
    #print(theta1[i], theta2[i], theta3[i])
    coordinates = a.forward(theta1[i], theta2[i], theta3[i])
    x_points.append(coordinates[0])
    y_points.append(coordinates[1])
    z_points.append(coordinates[2])

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_points, y_points, z_points, c=z_points, marker='o')

# Add labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the interactive plot
plt.show()
