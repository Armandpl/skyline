import math
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled, InvalidAction
from pygame import gfxdraw
from shapely.geometry import Point

from trajectory_optimization.bicycle_model import Car
from trajectory_optimization.track import Track

VIDEO_W = 500
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

FPS = 40  # Frames per second


class CarRacing(gym.Env):
    """## Description

    ## Action Space
    If continuous there are 3 actions :
    - 0: steering, -1 is full left, +1 is full right
    - 1: gas
    - 2: breaking

    ## Observation Space

    TODO

    ## Rewards
    TODO

    ## Starting State
    TODO

    ## Episode Termination
    TODO

    ## Arguments
    TODO


    ## Reset Arguments
    TODO

    ## TODO Credits
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        nb_rays: int = 13,
        random_init: bool = True,
        crash_penalty_weight: float = 1,
        # TODO
        # track id
        # car params?
        # randomize car param? randomize init state? randomize track
        # track and init state make sense, car don't?
        # terminate on lap?
    ):
        # TODO lap time when lap is completed

        self.random_init = random_init
        self.crash_penalty_weight = crash_penalty_weight

        self.action_space = spaces.Box(
            low=np.array([-1, -1]).astype(np.float32),
            high=np.array([+1, +1]).astype(np.float32),
        )

        # range finder + speed TODO anything else? track curvature? distance to centerline? angle with centerline?
        # we do need smth about we way the car is going, CW or CCW
        # bc reward will be negative if the car goes the wrong way but drives well
        # could also just make the progress reward absolute
        # this makes it possible for the car to reward hack and go in circles really fast
        # but I think the track isn't wide enough to go in circles soooo
        self.observation_space = spaces.Box(
            np.array([-np.inf] * (nb_rays + 1)).astype(np.float32),
            np.array([np.inf] * (nb_rays + 1)).astype(np.float32),
        )

        self.track = Track()
        self.rays = np.linspace(-90, 90, nb_rays)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def car_inside(self):
        """iterate over car vertices and check if they are inside the track."""
        for x, y in self.car.vertices:
            if not self.track.is_inside(x, y):
                return False

        return True

    def _get_obs(self):
        self.lidar = self.track.get_distance_to_side(
            self.car.pos_x, self.car.pos_y, self.car.yaw, self.rays
        )
        observation = np.array([self.car.speed, *self.lidar.values()], dtype=np.float32)
        return observation

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None

        self.steps_in_lap += 1

        progress = self.track.get_progress(self.car.pos_x, self.car.pos_y)
        self.car.step(action, 1 / FPS)

        # progress is measured from the starting/finish line
        # it is measured along the centerline
        new_progress = self.track.get_progress(self.car.pos_x, self.car.pos_y)

        # if points are on either side of the line it'll mess up the delta progress computation
        # if we cross the start line, the new progress is ~0 so we get:
        # delta_progress=abs(~0 - track.center.length) = ~track.center.length which is not the delta progress
        track_len = self.track.center.length

        # TODO double check this
        raw_delta_progress = new_progress - progress
        delta_progress = raw_delta_progress - track_len * round(raw_delta_progress / track_len)
        delta_progress = abs(delta_progress)

        self.total_progress += delta_progress

        step_reward = delta_progress
        terminated = False
        truncated = False

        if not self.car_inside():
            terminated = True
            # compute a crash penalty else the agent crashes the car
            crash_penalty = (self.car.speed**2) * self.crash_penalty_weight
            step_reward -= crash_penalty

        observation = self._get_obs()

        if self.render_mode == "human":
            self.render()

        car_pos = Point(self.car.pos_x, self.car.pos_y)
        distance_to_centerline = self.track.center.distance(car_pos)
        road_side = self.track.center_polygon.contains(
            car_pos
        )  # to know which side of the road we're driving on
        distance_to_centerline = distance_to_centerline if road_side else -distance_to_centerline

        info = {
            "distance_to_centerline": distance_to_centerline,
            "total_progress": self.total_progress,
            "pct_progress": self.total_progress / self.track.center.length * 100,
            "lap_time": None,
        }

        if self.total_progress >= self.track.center.length:  # lap completed
            lap_time = 1 / FPS * self.steps_in_lap
            info["lap_time"] = lap_time
            self.steps_in_lap = 0
            terminated = True

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

                self.car = Car(initial_state=init_state)

                if self.car_inside():
                    break

            # init speed
            # TODO the way we get model params in different places is not super consistent
            v_x = np.random.uniform(self.car.min_speed, self.car.max_speed)
            self.car.state[3] = v_x
        else:
            # TODO x, and y work for vivatech_2023.dxf but might not work for other track
            # they might be outside and will make the episode terminate instantly
            init_state = np.array([1.0, 0.5, 1e-9, 1e-9, 1e-9, 1e-9])
            self.car = Car(initial_state=init_state)

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

        # scale car position
        pos_x = self.car.pos_x * self.render_scale
        pos_y = self.car.pos_y * self.render_scale

        # render the lidar measurements as lines from the car position
        for angle, distance in self.lidar.items():
            # Calculate the end point of the line based on car's position, orientation, and lidar measurements
            end_x = self.car.pos_x * self.render_scale + distance * self.render_scale * np.cos(
                self.car.yaw + angle
            )
            end_y = self.car.pos_y * self.render_scale + distance * self.render_scale * np.sin(
                self.car.yaw + angle
            )

            # Draw the lidar line using pygame
            pygame.draw.line(self.surf, (0, 255, 0), (pos_x, pos_y), (end_x, end_y), 1)

        # Draw the car's rectangle on the screen using pygame
        pygame.draw.polygon(self.surf, (255, 255, 255), global_vertices)

        # pygame plumbing
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CarRacing(render_mode="human")

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
