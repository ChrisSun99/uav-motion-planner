"""
This file is not used for grading at all, and you should modify it any way you find useful.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from flightsim.animate import animate
from flightsim.simulate import Quadrotor, simulate
from flightsim.world import World
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim import hover_traj

import waypoint_traj
import se3_control
from occupancy_map import OccupancyMap
import os

from rrt3d_modified import *
import argparse

parser = argparse.ArgumentParser(description='Accepts various command line arguments.')
parser.add_argument('-s', '--save_to_disk', action='store_true',
                    help="Save the animation to disk at ./data/out.mp4.", default=True)
parser.add_argument('-e', '--empty_world', action='store_true',
                    help="Create an empty world instead of from a pre-configured file.")
parser.add_argument('-f', '--world_file', type=str, default='worlds/test_maze.json',
                    help='World config JSON file location')
p = parser.parse_args()

# This object defines the quadrotor dynamical model and should not be changed.
quadrotor = Quadrotor(quad_params)

# You will complete the implementation of the SE3Control object.
my_se3_control = se3_control.SE3Control(quad_params)

# This simple hover trajectory is useful for tuning control gains.
# my_traj = hover_traj.HoverTraj()

# Set simulation parameters.
#
# You may use the initial condition and a simple hover trajectory to examine the
# step response of your controller to an initial disturbance in position or
# orientation.

w = 6.0
world = World.from_file(p.world_file) if not p.empty_world else World.empty((0, w, 0, w, 0, w))

t_final = 30
initial_state = {'x': np.array([0, 0, 0]),
                 'v': np.zeros(3,),
                 'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
                 'w': np.zeros(3,)}

# define goal location
# for original map 
goal_location = np.array([5, 5, 5]) 


# For maze, default 
goal_location = np.array([9.0, 7.0, 1.5]) 
initial_state['x'] = np.array([1.0, 5.0, 1.5])

# using rrt to generate waypoints 
points = rrt_search(world, initial_state['x'], goal_location, resolution=10, rrt_star=True)


# Hand code the trajectory if necessary

my_traj_noop = waypoint_traj.WaypointTraj(points, total_T=14, optimize_time=False)
my_traj = waypoint_traj.WaypointTraj(points, total_T=14, optimize_time=True)

print("Reach time before optimization:")
print(my_traj_noop.reach_time)
print("Reach time after optimization:")
print(my_traj.reach_time)

# Perform simulation.
#
# This function performs the numerical simulation.  It returns arrays reporting
# the quadrotor state, the control outputs calculated by your controller, and
# the flat outputs calculated by you trajectory.

# print('Simulate without time optimized.')
# (time_noop, state_noop, control_noop, flat_noop, exit_noop) = simulate(initial_state,
#                                                                        quadrotor,
#                                                                        my_se3_control,
#                                                                        my_traj_noop,
#                                                                        t_final)
# print(exit_noop.value)

print('Simulate with time optimized.')
(time, state, control, flat, exit) = simulate(initial_state,
                                              quadrotor,
                                              my_se3_control,
                                              my_traj,
                                              t_final)
print(exit.value)

# Plot Results
#
# You will need to make plots to debug your controllers and tune your gains.
# Here are some example of plots that may be useful.

# Position and Velocity vs. Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time')
x = state['x']
x_des = flat['x']
ax = axes[0]
ax.plot(time, x_des[:, 0], 'r', time, x_des[:, 1], 'g', time, x_des[:, 2], 'b')
ax.plot(time, x[:, 0], 'r.', time, x[:, 1], 'g.', time, x[:, 2], 'b.')
ax.legend(('x', 'y', 'z'))
ax.set_ylabel('position, m')
ax.grid('major')
ax.set_title('Position')
v = state['v']
v_des = flat['x_dot']
ax = axes[1]
ax.plot(time, v_des[:, 0], 'r', time, v_des[:, 1], 'g', time, v_des[:, 2], 'b')
ax.plot(time, v[:, 0], 'r.', time, v[:, 1], 'g.', time, v[:, 2], 'b.')
ax.legend(('x', 'y', 'z'))
ax.set_ylabel('velocity, m/s')
ax.set_xlabel('time, s')
ax.grid('major')

# Orientation and Angular Velocity vs. Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time')
q_des = control['cmd_q']
q = state['q']
ax = axes[0]
ax.plot(time, q_des[:, 0], 'r', time, q_des[:, 1], 'g', time, q_des[:, 2], 'b', time, q_des[:, 3], 'k')
ax.plot(time, q[:, 0], 'r.', time, q[:, 1], 'g.', time, q[:, 2], 'b.', time, q[:, 3], 'k.')
ax.legend(('i', 'j', 'k', 'w'))
ax.set_ylabel('quaternion')
ax.set_xlabel('time, s')
ax.grid('major')
w = state['w']
ax = axes[1]
ax.plot(time, w[:, 0], 'r.', time, w[:, 1], 'g.', time, w[:, 2], 'b.')
ax.legend(('x', 'y', 'z'))
ax.set_ylabel('angular velocity, rad/s')
ax.set_xlabel('time, s')
ax.grid('major')

# Commands vs. Time
(fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Commands vs Time')
s = control['cmd_motor_speeds']
ax = axes[0]
ax.plot(time, s[:, 0], 'r.', time, s[:, 1], 'g.', time, s[:, 2], 'b.', time, s[:, 3], 'k.')
ax.legend(('1', '2', '3', '4'))
ax.set_ylabel('motor speeds, rad/s')
ax.grid('major')
ax.set_title('Commands')
M = control['cmd_moment']
ax = axes[1]
ax.plot(time, M[:, 0], 'r.', time, M[:, 1], 'g.', time, M[:, 2], 'b.')
ax.legend(('x', 'y', 'z'))
ax.set_ylabel('moment, N*m')
ax.grid('major')
T = control['cmd_thrust']
ax = axes[2]
ax.plot(time, T, 'k.')
ax.set_ylabel('thrust, N')
ax.set_xlabel('time, s')
ax.grid('major')

# 3D Paths
fig = plt.figure('3D Path')
ax = Axes3Ds(fig)
world.draw(ax)
ax.plot3D(state['x'][:, 0], state['x'][:, 1], state['x'][:, 2], 'b.')
ax.plot3D(flat['x'][:, 0], flat['x'][:, 1], flat['x'][:, 2], 'r')
# ax.plot3D(flat_noop['x'][:, 0], flat_noop['x'][:, 1], flat_noop['x'][:, 2], 'k')

# Animation (Slow)
# Instead of viewing the animation live, you may provide a .mp4 filename to save.
R = Rotation.from_quat(state['q']).as_matrix()
print(state['x'][-1])
if p.save_to_disk:
    os.makedirs('data', exist_ok=True)
    ani = animate(time, state['x'], R, world=world, filename="data/out.mp4")
else:
    ani = animate(time, state['x'], R, world=world)

plt.show()
