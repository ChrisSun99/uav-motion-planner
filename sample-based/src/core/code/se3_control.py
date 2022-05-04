import numpy as np
from scipy.spatial.transform import Rotation


class SE3Control(object):
    """

    """

    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.arm_length = quad_params['arm_length']  # meters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2

        # STUDENT CODE HERE

        # Controller parameter is defined here
        self.K_d = 0.5 * np.array([4.2, 4.2, 3.5])
        self.K_p = 0.5 * np.array([3.6, 3.6, 3.5])
        self.K_R = 150 * np.array([1.0, 1.0, 0.1])
        self.K_omega = 30 * np.array([1.0, 1.0, 1.0])

        # define force matrix
        self.Force_matrix = np.array([[self.k_thrust, self.k_thrust, self.k_thrust, self.k_thrust],
                                      [0, self.k_thrust * self.arm_length, 0, -self.k_thrust * self.arm_length],
                                      [-self.k_thrust * self.arm_length, 0, self.k_thrust * self.arm_length, 0],
                                      [self.k_drag, -self.k_drag, self.k_drag, -self.k_drag]])

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE

        # unpack input dict
        x_dot = state['v']
        x = state['x']
        q = state['q']
        omega = state['w']
        R = Rotation.from_quat(q).as_matrix()
        x_ddot_T = flat_output['x_ddot']
        x_dot_T = flat_output['x_dot']
        x_T = flat_output['x']
        psi_T = flat_output['yaw']
        omega_T = np.zeros(3)  # TODO: change it to output of the trajectory generator z_T

        # calculate u_1
        r_ddot_des = x_ddot_T - self.K_d * (x_dot - x_dot_T) - self.K_p * (x - x_T)
        F_des = self.mass * r_ddot_des + np.array([0.0, 0.0, self.mass * self.g])
        b_3 = R @ np.array([0, 0, 1])
        u_1 = b_3.T @ F_des

        # calculate u_2
        b_3_des = F_des / np.linalg.norm(F_des)
        a_psi = np.array([np.cos(psi_T), np.sin(psi_T), 0])
        b_2_des = np.cross(b_3_des, a_psi) / np.linalg.norm(np.cross(b_3_des, a_psi))
        R_des = np.hstack((np.cross(b_2_des, b_3_des).reshape((3, 1)),
                           b_2_des.reshape((3, 1)),
                           b_3_des.reshape((3, 1))))
        e_R_matrix = R_des.T @ R - R.T @ R_des
        e_R_vector = 1 / 2 * np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])
        u_2 = self.inertia @ (-self.K_R * e_R_vector - self.K_omega * (omega - omega_T))

        # compute motor_speed input
        u = np.concatenate((np.expand_dims(u_1, axis=0), u_2))
        cmd_motor_speeds = np.sqrt(np.linalg.inv(self.Force_matrix) @ u)
        cmd_thrust = u_1
        cmd_moment = u_2

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q}
        return control_input
