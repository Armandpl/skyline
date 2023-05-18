#!/usr/bin/env python
"""
@author: edwardahn

Class defining kinematics and dynamics of a RWD vehicle.

Code based on MATLAB simulation code written by Emily Yunan, located
at https://github.com/jsford/FFAST.
"""

import numpy as np
import yaml
from scipy.integrate import solve_ivp

from trajectory_optimization import data_dir


class Car:
    def __init__(
        self,
        model_type="linear",
        mu_s=1.37,
        mu_k=1.96,
        initial_state=None,
        model_path=data_dir / "models/fazer_mk2.yaml",
    ):
        self.model_type = model_type
        self.model_path = model_path

        # init state can't be strictly zeros else computation errors
        self.initial_state = initial_state if initial_state is not None else np.array([1e-6] * 6)
        self.mu_s = mu_s  # TODO load that from yaml? what is it?
        self.mu_k = mu_k
        self.load_parameters()

    def load_parameters(self):
        # TODO use hydra instead of loading and indexing the yaml
        with open(self.model_path) as f:
            params = yaml.safe_load(f)
        if self.model_type == "linear":
            self.model = LinearTireModel(params, self.mu_s, self.mu_k)
        elif self.model_type == "brush":
            self.model = BrushTireModel(params, self.mu_s, self.mu_k)
        else:
            raise ValueError("Invalid model type")

        self.max_steer = np.deg2rad(params["max_steer"])
        self.max_speed = params["max_speed"]
        self.min_speed = 0.85  # m/s, min speed else integrating the accel is too slow. TODO this depends on the model param (and probably on the compute as well)
        # TODO maybe there is a cleaner way to fix this? understand why integrating can be slow sometimes? division by zero or smth?
        self.max_accel = params["max_accel"]
        if params["max_steering_rate"] is not None:
            self.max_steering_rate = np.deg2rad(params["max_steering_rate"])
        else:
            self.max_steering_rate = None
        self.steering = 0
        self.width = params["body_w"]
        self.length = self.model.L
        self.state = self.initial_state

    def step(self, U, dt):
        """U is the action coming from the gym env, U[0] is delta, U[1] is speed U[0] is between -1
        and 1 U[1] is between 0 and 1."""
        # TODO maybe don't clip and rescale since sb3 already does it?
        U = np.copy(U)
        U[0] = np.clip(U[0], -1, 1) * self.max_steer  # steering command
        U[1] = (
            np.clip(U[1], 0, 1) * (self.max_speed - self.min_speed) + self.min_speed
        )  # speed command

        if self.max_accel is not None:
            speed_diff = U[1] - self.speed  # desired speed diff
            # assume we can speed up or slow down with the same limits
            speed_diff = np.clip(speed_diff, -self.max_accel * dt, self.max_accel * dt)
            U[1] = self.speed + speed_diff

        if self.max_steering_rate is not None:
            steering_diff = U[0] - self.steering
            steering_diff = np.clip(
                steering_diff, -self.max_steering_rate * dt, self.max_steering_rate * dt
            )

            self.steering += steering_diff
            U[0] = self.steering

        self.state = self.model.state_transition(self.state, U, dt)

    @property
    def vertices(self):
        # Calculate the car's rectangle vertices based on its position and orientation
        half_width = self.width / 2
        L_f, L_r = self.model.L_f, self.model.L_r

        # Define the car's rectangle vertices in the local coordinate system
        local_vertices = [
            (-L_r, -half_width),
            (-L_r, half_width),
            (L_f, half_width),
            (L_f, -half_width),
        ]

        # Rotate and translate the vertices to the global coordinate system
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)

        # scale from real world meters to pixels
        pos_x = self.pos_x
        pos_y = self.pos_y

        global_vertices = [
            (pos_x + cos_yaw * vx - sin_yaw * vy, pos_y + sin_yaw * vx + cos_yaw * vy)
            for vx, vy in local_vertices
        ]
        return global_vertices

    @property
    def pos_x(self):
        return self.state[0]

    @property
    def pos_y(self):
        return self.state[1]

    @property
    def yaw(self):
        return self.state[2]

    @property
    def v_x(self):
        return self.state[3]

    @property
    def v_y(self):
        return self.state[4]

    @property
    def speed(self):
        return np.sqrt(np.power(self.v_x, 2) + np.power(self.v_y, 2))

    @property
    def yaw_rate(self):
        return self.state[5]


class DynamicBicycleModel:
    """Vehicle modeled as a three degrees of freedom dynamic bicycle model."""

    def __init__(self, params, mu_s=1.37, mu_k=1.96):
        """Initialize model parameters from dictionary format to instance variables."""
        # Vehicle parameters
        self.m = params["m"]  # Mass
        self.L_f = params["L_f"]  # CoG to front axle length
        self.L_r = params["L_r"]  # CoG to rear axle length
        self.L = self.L_f + self.L_r  # Front to rear axle length
        self.load_f = params["load_f"]  # Load on front axle
        self.load_r = params["load_r"]  # Load on rear axle

        # Wheel parameters
        self.C_x = params["C_x"]  # Longitudinal stiffness
        self.C_alpha = params["C_alpha"]  # Cornering stiffness
        self.I_z = params["I_z"]  # Moment of inertia

        # Static and kinetic coefficients of friction
        self.mu_s = mu_s
        self.mu_k = mu_k

        # Slip ratio
        self.kappa = None

    def state_transition(self, X, U, dt):
        """Update state after some timestep."""
        t = np.array([0, dt])

        X_new = solve_ivp(fun=(lambda t, X: self._dynamics(X, t, U)), t_span=t, y0=X, atol=1e-5)
        return X_new.y[:, -1]

    def _dynamics(self, X, t, U):
        """Use dynamics model to compute X_dot from X, U."""
        pos_x = X[0]
        pos_y = X[1]
        pos_yaw = DynamicBicycleModel.wraptopi(X[2])
        v_x = X[3]
        v_y = X[4]
        yaw_rate = X[5]
        cmd_vx = U[1]  # m/s ?
        delta = U[0]  # rad ?

        # Tire slip angle (zero when stationary)
        if np.abs(v_x) < 0.01 and np.abs(v_y) < 0.01:
            alpha_f = 0
            alpha_r = 0
        else:
            alpha_f = np.arctan2((v_y + self.L_f * yaw_rate), v_x) - delta
            alpha_r = np.arctan2((v_y - self.L_r * yaw_rate), v_x)

        # Compute forces on tires using brush tire model
        F_yf = self._tire_dynamics_front(alpha_f)
        F_xr, F_yr = self._tire_dynamics_rear(v_x, cmd_vx, alpha_r)

        # Find dX
        T_z = self.L_f * F_yf * np.cos(delta) - self.L_r * F_yr
        ma_x = F_xr - F_yf * np.sin(delta)
        ma_y = F_yf * np.cos(delta) + F_yr

        # Acceleration with damping
        yaw_rate_dot = T_z / self.I_z - 0.02 * yaw_rate
        v_x_dot = ma_x / self.m + yaw_rate * v_y - 0.025 * v_x
        v_y_dot = ma_y / self.m - yaw_rate * v_x - 0.025 * v_y

        # Translate to inertial frame
        v = np.sqrt(v_x**2 + v_y**2)
        beta = np.arctan2(v_y, v_x)
        pos_x_dot = v * np.cos(beta + pos_yaw)
        pos_y_dot = v * np.sin(beta + pos_yaw)

        X_dot = np.zeros(6)
        X_dot[0] = pos_x_dot
        X_dot[1] = pos_y_dot
        X_dot[2] = yaw_rate
        X_dot[3] = v_x_dot
        X_dot[4] = v_y_dot
        X_dot[5] = yaw_rate_dot

        return X_dot

    def _tire_dynamics_front(self, alpha):
        """
        :param alpha: Slip angle
        :return: lateral force on the front tires
        """
        raise NotImplementedError

    def _tire_dynamics_rear(self, v_x, wheel_vx, alpha):
        """
        :param v_x: Current longitudinal velocity
        :param wheel_vx: Commanded wheel velocity
        :param alpha: Slip angle
        :return: longitudinal and lateral forces on the rear tires
        """
        raise NotImplementedError

    @staticmethod
    def wraptopi(val):
        """Wrap radian value to the interval [-pi, pi]."""
        pi = np.pi
        val = val - 2 * pi * np.floor((val + pi) / (2 * pi))
        return val


class LinearTireModel(DynamicBicycleModel):
    """Use a dynamic bicycle model with a linear tire model for tire dynamics."""

    def __init__(self, params, mu_s, mu_k):
        super().__init__(params, mu_s, mu_k)

    def _tire_dynamics_front(self, alpha):
        F_yf = -self.C_alpha * alpha
        return F_yf

    def _tire_dynamics_rear(self, v_x, wheel_vx, alpha):
        self.kappa = (wheel_vx - v_x) / v_x
        F_xr = self.C_x * self.kappa
        F_yr = -self.C_alpha * alpha
        return F_xr, F_yr


class BrushTireModel(DynamicBicycleModel):
    """Use a dynamic bicycle model with a brush tire model for tire dynamics."""

    def __init__(self, params, mu_s, mu_k):
        super().__init__(params, mu_s, mu_k)

    def _tire_dynamics_front(self, alpha):
        # alpha > pi/2 is invalid because of the use of tan(). Since
        # alpha > pi/2 means vehicle moving backwards, Fy's sign has
        # to be reversed, hence we multiply by sign(alpha)
        if abs(alpha) > np.pi / 2:
            alpha = (np.pi - abs(alpha)) * np.sign(alpha)

        # Compute slip angle where total sliding occurs alpha_sl
        alpha_sl = np.arctan(3 * self.mu_s * self.load_f / self.C_alpha)

        if abs(alpha) <= alpha_sl:
            tan = np.tan(alpha)
            first = -self.C_alpha * tan
            second = self.C_alpha**2 / (3 * self.mu_s * self.load_f) * np.abs(tan) * tan
            third = -self.C_alpha**3 / (27 * self.mu_s**2 * self.load_f**2) * tan**3
            Fy = first + second + third
        else:
            Fy = -self.mu_s * self.load_f * np.sign(alpha)

        return Fy

    def _tire_dynamics_rear(self, v_x, wheel_vx, alpha):
        # Find longitudinal wheel slip K (kappa)
        if np.abs(wheel_vx - v_x) < 0.01 or (np.abs(wheel_vx) < 0.01 and np.abs(v_x) < 0.01):
            K = 0
        # Infinite slip, longitudinal saturation
        elif abs(v_x) < 0.01:
            Fx = np.sign(wheel_vx) * self.mu_s * self.load_r
            Fy = 0
            return Fx, Fy
        else:
            K = (wheel_vx - v_x) / np.abs(v_x)
        self.kappa = K

        # Instead of avoiding -1, now look for positive equivalent
        if K < 0:
            spin_dir = -1
            K = np.abs(K)
        else:
            spin_dir = 1

        # alpha > pi/2 is invalid because of the use of tan(). Since
        # alpha > pi/2 means vehicle moving backwards, Fy's sign has
        # to be reversed, hence we multiply by sign(alpha)
        if abs(alpha) > np.pi / 2:
            alpha = (np.pi - abs(alpha)) * np.sign(alpha)

        # Compute combined slip value gamma
        gamma = np.sqrt(
            self.C_x**2 * (K / (1 + K)) ** 2 + self.C_alpha**2 * (np.tan(alpha) / (1 + K)) ** 2
        )

        if gamma <= 3 * self.mu_s * self.load_r:
            F = (
                gamma
                - 1 / (3 * self.mu_s * self.load_r) * gamma**2
                + 1 / (27 * self.mu_s**2 * self.load_r**2) * gamma**3
            )
        else:
            F = self.mu_k * self.load_r

        if gamma == 0:
            Fx = 0
            Fy = 0
        else:
            Fx = self.C_x / gamma * (K / (1 + K)) * F * spin_dir
            Fy = -self.C_alpha / gamma * (np.tan(alpha) / (1 + K)) * F

        return Fx, Fy
