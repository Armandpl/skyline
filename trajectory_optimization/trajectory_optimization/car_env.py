from typing import Optional, Union

import gymnasium as gym
import hydra
import numpy as np
import pygame
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf
from shapely.geometry import Point

from trajectory_optimization import data_dir
from trajectory_optimization.track import Track
from trajectory_optimization.wrappers import RescaleWrapper

VIDEO_W = 500
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


class CarRacing(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
    }

    def __init__(
        self,
        car_config: DictConfig,
        track="tracks/vivatech_2023.dxf",
        track_obstacles="tracks/vivatech_2023_obstacles.dxf",
        render_mode: Optional[str] = "rgb_array",
        nb_rays: int = 13,
        max_lidar_distance: float = 15,  # max lidar distance, in m
        random_init: bool = True,
        fps: int = 30,  # fps == dt
        fixed_speed: Optional[float] = None,
        max_wheels_out: int = 2,
        # TODO
        # track id
        # car params?
        # randomize car param? randomize track?
    ):
        self.car_config = car_config
        self.random_init = random_init
        self.fixed_speed = fixed_speed
        self.max_wheels_out = max_wheels_out
        self.max_lidar_distance = max_lidar_distance

        max_steer_rad = np.deg2rad(car_config.max_steer)
        if self.fixed_speed is None:
            self.action_space = spaces.Box(
                low=np.array([-max_steer_rad, car_config.min_speed]).astype(np.float32),
                high=np.array([+max_steer_rad, car_config.max_speed]).astype(np.float32),
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-max_steer_rad]).astype(np.float32),
                high=np.array([+max_steer_rad]).astype(np.float32),
            )

        self.track = Track(
            filepath=data_dir / track, obstacles_filepath=data_dir / track_obstacles
        )

        # add a lidar for obstacles if there obstacles on the track
        # nb_rays + 1 if no obstacles, nb_rays * 2 + 1 if obstacles
        observation_shape = nb_rays * (1 + 1 * (len(self.track.obstacles) > 0)) + 1
        self.observation_space = spaces.Box(
            low=np.array([-1] * observation_shape).astype(np.float32),
            high=np.array([+1] * observation_shape).astype(np.float32),
        )

        self.rays = np.linspace(-90, 90, nb_rays)

        self.render_mode = render_mode
        self.fps = fps
        self.screen = None
        self.clock = None

    def _is_outside(self) -> bool:
        """car is outside if it's center of mass is outside or if it has > self.max_wheels_out
        wheels outside the track."""
        if not self.track.is_inside(self.car.pos_x, self.car.pos_y):
            return True

        if self.max_wheels_out < 4:
            outside = 0
            for x, y in self.car.vertices:
                if not self.track.is_inside(x, y):
                    outside += 1
                    if outside > self.max_wheels_out:
                        return True
        return False

    def _is_crashed(self) -> bool:
        """check if any of the car vertices is inside an obstacle."""
        for x, y in self.car.vertices:
            for obstacle in self.track.obstacles:
                if obstacle.contains(Point(x, y)):
                    return True
        return False

    def _get_obs(self):
        self.lidar = self.track.get_distances_to_sides(
            self.car.pos_x, self.car.pos_y, self.car.yaw, self.rays
        )

        if len(self.track.obstacles) > 0:
            self.obstacles_lidar = self.track.get_distances_to_obstacles(
                self.car.pos_x, self.car.pos_y, self.car.yaw, self.rays
            )
            observation = np.array(
                [self.car.speed, *self.lidar.values(), *self.obstacles_lidar.values()],
                dtype=np.float32,
            )
        else:
            observation = np.array([self.car.speed, *self.lidar.values()], dtype=np.float32)

        return observation

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None

        self.steps_in_lap += 1
        info = {}

        # progress is measured from the starting/finish line
        # it is measured along the centerline
        progress = self.track.get_progress(self.car.pos_x, self.car.pos_y)
        self.car.step(action, 1 / self.fps)
        new_progress = self.track.get_progress(self.car.pos_x, self.car.pos_y)

        # TODO double check this
        # if points are on either side of the line it'll mess up the delta progress computation
        # if we cross the start line, the new progress is ~0 so we get:
        # delta_progress=abs(~0 - track.center.length) = ~track.center.length which is not the delta progress
        track_len = self.track.center.length
        raw_delta_progress = new_progress - progress
        delta_progress = raw_delta_progress - track_len * round(raw_delta_progress / track_len)
        delta_progress = abs(delta_progress)

        self.total_progress += (
            delta_progress  # keep track of progress to know when we completed a lap
        )

        step_reward = delta_progress
        truncated = False

        if self.total_progress >= self.track.center.length:  # lap completed
            lap_time = 1 / self.fps * self.steps_in_lap
            info["lap_time"] = lap_time
            self.steps_in_lap = 0
            terminated = True
        else:
            terminated = self._is_outside() or self._is_crashed()

        observation = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return observation, step_reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.total_progress = 0
        self.steps_in_lap = 0

        # scale render based on track dimensions
        track_x_min, track_y_min, track_x_max, track_y_max = self.track.outer.bounds
        w_scale = WINDOW_W / track_x_max  # assume track is at origin, if not it'll be tiny
        h_scale = WINDOW_H / track_y_max

        self.render_scale = w_scale if w_scale <= h_scale else h_scale

        if self.random_init:
            while True:
                x = np.random.uniform(track_x_min, track_x_max)
                y = np.random.uniform(track_y_min, track_y_max)
                yaw = np.random.uniform(-np.pi, np.pi)

                init_state = np.array([x, y, yaw, 1e-9, 1e-9, 1e-9])

                self.car = hydra.utils.instantiate(
                    self.car_config, initial_state=init_state, fixed_speed=self.fixed_speed
                )
                # self.car = Car(initial_state=init_state, fixed_speed=self.fixed_speed)

                if not (self._is_outside() or self._is_crashed()):
                    break

            # init speed
            if self.fixed_speed is None:
                v_x = np.random.uniform(self.car.min_speed, self.car.max_speed)
                self.car.set_speed(v_x)
        else:
            # TODO x, and y work for vivatech_2023.dxf but might not work for other tracks
            # they might be outside and will make the episode terminate instantly
            # fix could be to always randomly spawn the car
            init_state = np.array([1.0, 0.5, 1e-9, 1e-9, 1e-9, 1e-9])
            self.car = hydra.utils.instantiate(
                self.car_config, initial_state=init_state, fixed_speed=self.fixed_speed
            )
            # self.car = Car(initial_state=init_state, fixed_speed=self.fixed_speed)

        return self._get_obs(), {}

    def render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        # render the track
        for line in [self.track.inner, self.track.outer]:
            # Transform track points to screen points and scale
            points = [(self.render_scale * p[0], self.render_scale * p[1]) for p in line.coords]

            # Draw lines with pygame
            pygame.draw.lines(self.surf, (255, 255, 255), False, points, 2)

        global_vertices = self.car.vertices
        global_vertices = [
            (x * self.render_scale, y * self.render_scale) for x, y in global_vertices
        ]

        # render the track obstacles
        for obstacle in self.track.obstacles:
            # Transform track points to screen points and scale
            points = [
                (self.render_scale * p[0], self.render_scale * p[1])
                for p in obstacle.exterior.coords
            ]

            # Draw lines with pygame
            pygame.draw.lines(self.surf, (255, 0, 0), False, points, 2)

        if len(self.track.obstacles) > 0:
            # show obstacle ray when they hit obstacle else show normal ray
            filtered_obstacle_lidar = {
                k: v for k, v in self.obstacles_lidar.items() if v != np.inf
            }
            self._render_lidar(filtered_obstacle_lidar, self.surf, (0, 0, 255), 1)
            filtered_lidar = {
                k: v for k, v in self.lidar.items() if k not in filtered_obstacle_lidar.keys()
            }
            self._render_lidar(filtered_lidar, self.surf, (0, 255, 0), 1)
        else:
            self._render_lidar(self.lidar, self.surf, (0, 255, 0), 1)

        # Draw the car's rectangle on the screen using pygame
        pygame.draw.polygon(self.surf, (255, 255, 255), global_vertices)

        # pygame plumbing
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.fps)
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))

    def _render_lidar(self, lidar, surf, color, line_thickness=1):
        # scale car position
        pos_x = self.car.pos_x * self.render_scale
        pos_y = self.car.pos_y * self.render_scale

        # render the lidar measurements as lines from the car position
        for angle, distance in lidar.items():
            # Calculate the end point of the line based on car's position, orientation, and lidar measurements
            end_x = self.car.pos_x * self.render_scale + distance * self.render_scale * np.cos(
                self.car.yaw + angle
            )
            end_y = self.car.pos_y * self.render_scale + distance * self.render_scale * np.sin(
                self.car.yaw + angle
            )

            # Draw the lidar line using pygame
            pygame.draw.line(surf, color, (pos_x, pos_y), (end_x, end_y), line_thickness)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, -1.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0.0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0.0
                if event.key == pygame.K_UP:
                    a[1] = -1.0

            if event.type == pygame.QUIT:
                quit = True

    car_config = OmegaConf.load("../scripts/configs/env/car_config/base_car.yaml")
    env = CarRacing(car_config, render_mode="human")
    env = RescaleWrapper(env)

    quit = False
    while not quit:
        env.reset()
        env.render()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 100 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
                print(f"info: {info}")

            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
