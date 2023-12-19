from kinematics.kinematics import KinematicsSolver
from joint_trajectories import SimpleJointSpaceTrajectory
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

# Wszystkie wartości translacji w metrach
# Translacja T1 - pomiędzy J1, a J2
t1 = np.array([0.05, 0, -0.035, 1])

# Translacja T2 - pomiędzy J2, a J3
t2 = np.array([0.09, 0, 0, 1])

# Translacja T3 - pomiędzy J3, a FCP
t3 = np.array([0.1, 0, 0, 1])


kinematics_solver = KinematicsSolver(t1, t2, t3, 0, -np.pi/2)
trajectory = SimpleJointSpaceTrajectory(kinematics_solver)

start_coordinates = np.array([0, -0.02, -0.15])
end_coordinates = np.array([0, 0.02, -0.135])

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
    coordinates = kinematics_solver.forward(theta1[i], theta2[i], theta3[i])
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
