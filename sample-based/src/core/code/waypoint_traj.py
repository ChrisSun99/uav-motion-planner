import numpy as np
import cvxpy as cp
import copy


class WaypointTraj(object):
    """

    """

    def __init__(self, points: np.ndarray, total_T: float,
                 optimize_time: bool, use_min_snap: bool = True):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points,         (N, 3)  array of N waypoint coordinates in 3D
            total_T,        float   required total time for the drone to visit all waypoints
            optimize_time   bool    whether to optimize time (reallocate time) using constrained gradient descent
                                    in minimum-snap problem
        """
        np.random.seed(1)
        # define constants
        self.N = points.shape[0]
        self.total_T = total_T
        self.snap_poly_order = 7

        # store data
        self.points = points

        # initial guess of reach time is proportional to distance between two waypoints
        distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.reach_time = self.total_T / np.sum(distance) * distance

        # #### Another initial guess is to assign equal time duration to each spline segment
        # self.reach_time = np.array((self.N - 1) * [total_T / (self.N - 1)])

        self.use_min_snap = use_min_snap
        if use_min_snap:
            self.c_x = None
            self.c_y = None
            self.c_z = None

            if not optimize_time:
                # get spline parameters in all three axes
                #   Target polynomial: p(t) = c_7*t^7 + ... + c_1*t + c_0
                #   spline is 7-order in min-snap settings so c_x, c_y, c_z are expected to be
                #   of shape (8,)
                self.c_x, _ = WaypointTraj.get_spline_parameters(
                    self.points[:, 0], self.snap_poly_order, self.reach_time)
                self.c_y, _ = WaypointTraj.get_spline_parameters(
                    self.points[:, 1], self.snap_poly_order, self.reach_time)
                self.c_z, _ = WaypointTraj.get_spline_parameters(
                    self.points[:, 2], self.snap_poly_order, self.reach_time)
            else:
                # run segment time optimization
                self.c_x, self.c_y, self.c_z, self.reach_time =\
                    WaypointTraj.optimize_reach_time(self.reach_time, self.points, self.snap_poly_order)

    def update(self, t: float) -> dict:
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        # flat output initialization
        x = np.zeros(3)
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0.0
        yaw_dot = 0.0

        # calculate spline segment index & handle weird cases i.e. t == np.inf
        seg_index = np.searchsorted(np.cumsum(self.reach_time), t)

        # if segment index exceeds the last one,
        #   only populates the terminal point
        #
        # Trajectory need to specify waypoints where t == np.inf in order for
        #   simulator to terminate in normal mode
        #   --- "If None (default), terminate when hover is reached at
        #       the location of trajectory with t=inf."
        if seg_index >= self.N - 1:
            x = self.points[-1, :]
        elif self.use_min_snap:
            delta_t = t - np.insert(np.cumsum(self.reach_time), 0, 0)[seg_index]
            if delta_t == 0:
                vector_t = np.zeros(self.snap_poly_order + 1)
                vector_dt = np.zeros(self.snap_poly_order + 1)
                vector_ddt = np.zeros(self.snap_poly_order + 1)
                # the last element should be a constant, does not contain t
                #   so should never be 0
                vector_t[-1] = 1
                vector_dt[-1] = 1
                vector_ddt[-1] = 1
            else:
                # construction of [T^7, T^6, T^5, ..., T, 0]
                vector_t = np.array([delta_t ** (self.snap_poly_order - i) for i in range(self.snap_poly_order + 1)])
                # construction of [7T^6, 6T^5, 5T^4, ..., 1, 0]
                vector_dt = np.array([(self.snap_poly_order - i) * delta_t ** (self.snap_poly_order - i - 1)
                                      for i in range(self.snap_poly_order + 1)])
                # construction of [42T^5, 30T^4, 20T^3, ...,2, 0, 0]
                vector_ddt = np.array([(self.snap_poly_order - i - 1) * (self.snap_poly_order - i) *
                                       delta_t ** (self.snap_poly_order - i - 2)
                                       for i in range(self.snap_poly_order + 1)])

            x[0] = self.c_x[seg_index, :].T @ vector_t
            x[1] = self.c_y[seg_index, :].T @ vector_t
            x[2] = self.c_z[seg_index, :].T @ vector_t
            x_dot[0] = self.c_x[seg_index, :].T @ vector_dt
            x_dot[1] = self.c_y[seg_index, :].T @ vector_dt
            x_dot[2] = self.c_z[seg_index, :].T @ vector_dt
            x_ddot[0] = self.c_x[seg_index, :].T @ vector_ddt
            x_ddot[1] = self.c_y[seg_index, :].T @ vector_ddt
            x_ddot[2] = self.c_z[seg_index, :].T @ vector_ddt
        else:
            x = self.points[seg_index, :]

        # return flat output
        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot,
                       'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output

    @staticmethod
    def get_spline_parameters(points: np.ndarray, snap_poly_order: int, reach_time: np.ndarray) -> tuple:
        """Runs the minimum-snap optimization problem with given 1-D waypoint input & specified timesteps.

        Args:
            points (np.ndarray): (N,)           1-D waypoint input
            snap_poly_order (int):              spline order (min-snap problem is 7)
            reach_time (np.ndarray): (N-1,)     required time duration for each spline segment

        Returns:
            tuple: (spline_coefficients, optimal_value)
        """
        N = len(points)

        # define spline parameters
        c = cp.Variable((N - 1, snap_poly_order + 1))

        # initialize list of constraints
        list_constraints = []

        # initialize initial boundary conditions for each spline segment
        initial_location_constraint = np.array(snap_poly_order * [0] + [1])
        initial_velocity_constraint = np.array((snap_poly_order - 1) * [0] + [1] + [0])
        initial_accel_constraint = np.array((snap_poly_order - 2) * [0] + [2] + [0, 0])

        #   1. starting point: x(0) = x0, x_dot(0) = 0, x(T) = x1
        final_location_constraint = np.array([reach_time[0] ** (snap_poly_order - i)
                                              for i in range(snap_poly_order + 1)])
        M = np.vstack((initial_location_constraint, final_location_constraint,
                       initial_velocity_constraint))
        list_constraints += [M @ c[0, :] == np.array([points[0], points[1], 0])]

        #   2. middle points: x(iT) = xi, x(i(T + 1)) = xi+1
        for i in range(1, N - 2):
            # define boundary constraints
            final_location_constraint = np.array([reach_time[i] ** (snap_poly_order - j)
                                                  for j in range(snap_poly_order + 1)])
            final_velocity_constraint = np.array([(snap_poly_order - j) * reach_time[i - 1] ** (snap_poly_order - j - 1)
                                                  for j in range(snap_poly_order + 1)])
            final_accel_constraint = np.array(
                [(snap_poly_order - j - 1) * (snap_poly_order - j) * reach_time[i - 1] ** (snap_poly_order - j - 2)
                 for j in range(snap_poly_order + 1)])

            # construct boundary constraint matrix
            M = np.vstack((initial_location_constraint, final_location_constraint))
            #   continuity: location should be consistent, velocity & accel should be consistent
            #       between two consecutive segments
            list_constraints += [M @ c[i, :] == np.array([points[i], points[i + 1]])]
            list_constraints += [initial_velocity_constraint @ c[i, :] == final_velocity_constraint @ c[i - 1, :]]
            list_constraints += [initial_accel_constraint @ c[i, :] == final_accel_constraint @ c[i - 1, :]]

        #   3. final point: x((end - 1)T) = xend-1, x_dot(end) = 0, x(end * T) = xend
        final_location_constraint = np.array([reach_time[-1] ** (snap_poly_order - i)
                                              for i in range(snap_poly_order + 1)])
        final_velocity_constraint = np.array([(snap_poly_order - j) * reach_time[-1] ** (snap_poly_order - j - 1)
                                              for j in range(snap_poly_order + 1)])
        M = np.vstack((initial_location_constraint, final_location_constraint,
                       final_velocity_constraint))
        list_constraints += [M @ c[N - 2, :] == np.array([points[N - 2], points[N - 1], 0])]
        #   continuity: location should be consistent, velocity & accel should be consistent
        #       between two consecutive segments
        final_velocity_constraint = np.array([(snap_poly_order - j) * reach_time[-2] ** (snap_poly_order - j - 1)
                                              for j in range(snap_poly_order + 1)])
        final_accel_constraint = np.array(
            [(snap_poly_order - j - 1) * (snap_poly_order - j) * reach_time[-2] ** (snap_poly_order - j - 2)
             for j in range(snap_poly_order + 1)])
        list_constraints += [initial_velocity_constraint @ c[N - 2, :] == final_velocity_constraint @ c[N - 3, :]]
        list_constraints += [initial_accel_constraint @ c[N - 2, :] == final_accel_constraint @ c[N - 3, :]]

        # Construct objective function
        #   construct cost matrix
        objective_function = 0
        for i, T_i in enumerate(reach_time):
            # int_0^T (x_ddddot)^2 dt
            # ################### FOR REFERENCE; ORIGINAL FORMULA ###################
            # objective_function += 96 * reach_time * (5 * (reach_time ** 2) * (6 * (reach_time ** 2) * (35 * cp.square(c[i, 0]) * (reach_time ** 2) + 35 * c[i, 0] * c[i, 1] * reach_time + 9 * cp.square(c[i, 1]))
            #                                                                   + 3 * c[i, 2] * reach_time * (28 * c[i, 0] * reach_time + 15 * c[i, 1]) + 10 * cp.square(c[i, 2]))
            #                                          + 15 * c[i, 3] * reach_time * (reach_time * (7 * c[i, 0] * reach_time + 4 * c[i, 1]) + 2 * c[i, 2]) + 6 * cp.square(c[i, 3]))
            #########################################################################
            H = np.array([[100800 * T_i ** 7, 50400 * T_i ** 6, 20160 * T_i ** 5, 5040 * T_i ** 4],
                          [50400 * T_i ** 6, 25920 * T_i ** 5, 10800 * T_i ** 4, 2880 * T_i ** 3],
                          [20160 * T_i ** 5, 10800 * T_i ** 4, 4800 * T_i ** 3, 1440 * T_i ** 2],
                          [5040 * T_i ** 4, 2880 * T_i ** 3, 1440 * T_i ** 2, 576 * T_i]])
            objective_function += cp.quad_form(c[i, 0:4], H)

        # solve optimization problem
        prob = cp.Problem(cp.Minimize(objective_function), list_constraints)
        prob.solve()

        # return optimal solution
        return c.value, prob.value

    @staticmethod
    def optimize_reach_time(T0: np.ndarray, points: np.ndarray, snap_poly_order: int) -> np.ndarray:
        """Runs time segment optimization on the initial min-snap problem with constrained gradient descent.

        Args:
            T0 (np.ndarray): (N-1,)                                     time duration of each spline segment
            points (np.ndarray): (N,num_flat_output)                    waypoints
            snap_poly_order (int):                                      order of polynomial of each spline

        Returns:
            np.ndarray: (num_flat_output,N-1,snap_poly_order+1)     updated polynomial coefficients for each spline
        """
        # Use cache to store pre-computed f values
        #   to eliminate redundant computation
        table_f = dict()

        def f(points: np.ndarray, snap_poly_order: int, reach_time: np.ndarray):
            array_bytes = reach_time.data.tobytes()
            if array_bytes in table_f.keys():
                return table_f[array_bytes]
            else:
                n_dim = points.shape[1]
                ret = 0
                for i in range(n_dim):
                    ret += WaypointTraj.get_spline_parameters(points[:, i], snap_poly_order, reach_time)[1]
                table_f[array_bytes] = ret / n_dim
                return ret / n_dim

        print("----- Solving time optimization problem...")

        # define basis of gradient
        N = len(points)
        num_flat_output = points.shape[1]
        g_base = -1 / (N - 2) * np.ones(N - 1)

        # define small number for gradient computation
        h = 0.1

        # define params for backtracking line search
        alpha = 0.1
        beta = 0.5

        # define stopping criterion
        eps = 1e-8

        # define backtracking line search stopping criterion
        #   (time out threshold when main criterion is not satisfied)
        min_t_eps = 1e-10

        # initial variable
        T = T0
        T_SUM = T0.sum()

        # gradient descent main loop
        solve_timeout = False
        solve_min_accuracy_reached = False
        timeout = 10
        while True:
            # compute f(T)
            f_val = f(points, snap_poly_order, T)

            # calculate gradient of f
            grad_f = np.zeros_like(T0)
            for i in range(N - 1):
                # construct search direction
                g = copy.deepcopy(g_base)
                g[i] = 1
                # compute gradient component in this direction
                grad_f += 1 / h * (f(points, snap_poly_order, T + h * g) - f_val) * g

            # compute gradient descent step
            T_step = -1 * grad_f / np.linalg.norm(grad_f)

            # perform backtracking line search
            t = 1
            T_update = None
            while np.any(T + t * T_step < 0):
                t *= beta
            while True:
                T_update = T + t * T_step
                T_update *= T_SUM / np.sum(T_update)
                if f(points, snap_poly_order, T_update) < f_val + alpha * t * grad_f.T @ T_step:
                    break
                t *= beta
                # backtracking line search timeout condition
                if np.abs(t) < min_t_eps:
                    solve_min_accuracy_reached = True
                    break

            # stopping criterion
            if timeout <= 0 or np.linalg.norm(T_update - T) < eps:
                if timeout <= 0:
                    solve_timeout = True
                break

            # update
            T = T_update
            timeout -= 1

        if solve_min_accuracy_reached:
            print('----- Backtracking line search terminates early due to accuracy limit')
        if not solve_timeout:
            print("----- Solving successful!")
        else:
            print("----- Solve maximum time reached.")

        return [WaypointTraj.get_spline_parameters(points[:, i], snap_poly_order, T)[0]
                for i in range(num_flat_output)] + [T]


if __name__ == '__main__':
    points = np.array([
        [0, 0, 0],
        [2.5, 0, 0],
        [2.5, 5, 0],
        [0, 5, 0]])
    my_traj = WaypointTraj(points, total_T=8, optimize_time=True)
    my_traj_noop = WaypointTraj(points, total_T=8, optimize_time=False)

    print("Before time optimized:", my_traj_noop.reach_time)
    print("Time optimized:", my_traj.reach_time)
