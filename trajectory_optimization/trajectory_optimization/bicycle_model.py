#!/usr/bin/env python
"""
@author: edwardahn

Class defining kinematics and dynamics of a RWD vehicle.

Code based on MATLAB simulation code written by Emily Yunan, located
at https://github.com/jsford/FFAST.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from scipy.integrate import solve_ivp

from trajectory_optimization import data_dir


@dataclass
class DynamicBicycleModel:
    """Vehicle modeled as a three degrees of freedom dynamic bicycle model."""

    m: float  # Mass
    L_f: float  # CoG to front axle length
    L_r: float  # CoG to rear axle length
    load_f: float  # Load on front axle
    load_r: float  # Load on rear axle
    C_x: float  # Longitudinal stiffness
    C_alpha: float  # Cornering stiffness
    I_z: float  # Moment of inertia
    mu_s: float = 1.37  # static friction coef (depends on floor?)
    mu_k: float = 1.96  # kinetic friction coef
    kappa: float = None  # slip ratio

    @property
    def L(self):
        return self.L_f + self.L_r

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


class Car:
    """wrapper around bicycle model to:

    - load params
    - implement steering rate and max accel
    - have getters and setters for system state
    """

    def __init__(
        self,
        model: DynamicBicycleModel,
        max_steer: float,
        max_speed: float,
        max_accel: float,
        body_w: float,
        min_speed: float = 0.85,
        initial_state=None,
        max_steering_rate=None,
        fixed_speed: Optional[float] = None,  # only control the car steering
    ):
        self.model = model
        self.fixed_speed = fixed_speed

        # init state can't be strictly zeros else computation errors
        self.initial_state = initial_state if initial_state is not None else np.array([1e-6] * 6)

        self.max_steer = np.deg2rad(max_steer)
        self.max_speed = max_speed
        self.min_speed = min_speed  # m/s, min speed else integrating the accel is too slow. TODO this depends on the model param (and probably on the compute as well)
        # TODO maybe there is a cleaner way to fix this? understand why integrating can be slow sometimes? division by zero or smth?
        self.max_accel = max_accel
        if max_steering_rate is not None:
            self.max_steering_rate = np.deg2rad(max_steering_rate)
        else:
            self.max_steering_rate = None

        self.steering = 0
        self.width = body_w
        self.length = self.model.L
        self.state = self.initial_state

    def step(self, U, dt):
        """U[0] is delta in rad, U[1] is speed in m/s In the case of fixed speed, there is no speed
        command, only U[0] steering."""
        U = np.copy(
            U
        )  # else we modify the action sent by the RL algo and probably mess up its training data

        if self.max_accel is not None and self.fixed_speed is None:
            speed_diff = U[1] - self.speed  # desired speed diff
            # assume we can speed up or slow down with the same limits
            speed_diff = np.clip(speed_diff, -self.max_accel * dt, self.max_accel * dt)
            U[1] = self.speed + speed_diff

        if self.fixed_speed is not None:
            U = np.array([U[0], 0], dtype=np.float32)
            U[1] = self.fixed_speed

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
        """Calculate the car's rectangle vertices based on its position, orientation and body
        size."""

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

    def set_speed(self, v_x):
        self.state[3] = v_x

    @property
    def yaw_rate(self):
        return self.state[5]


class LinearTireModel(DynamicBicycleModel):
    """Use a dynamic bicycle model with a linear tire model for tire dynamics."""

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
