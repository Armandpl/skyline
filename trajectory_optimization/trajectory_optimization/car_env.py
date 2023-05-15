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

VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

# TODO compute scale based on track len
SCALE = 100  # meters to px, 10 m = 1000 px

FPS = 20  # Frames per second


class CarRacing(gym.Env):
    """## Description

    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```
    python gymnasium/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

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
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        nb_rays: int = 10,
        random_init: bool = True,
        # TODO
        # track id
        # car params?
        # randomize car param? randomize init state? randomize track
        #   track and init state make sense, car don't?
        # clockwise, counter clockwise?
    ):
        # TODO lap time when lap is completed

        self.random_init = random_init

        self.action_space = spaces.Box(
            np.array([-1, -1]).astype(np.float32),
            np.array([0, +1]).astype(np.float32),
        )  # steer, speed

        # TODO speed (x, y?), edge distance measurements
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

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None

        progress = self.track.get_progress(self.car.pos_x, self.car.pos_y)
        # TODO do i need to clip the action?
        self.car.step(action, 1 / FPS)
        delta_progress = self.track.get_progress(self.car.pos_x, self.car.pos_y) - progress

        step_reward = delta_progress
        terminated = False
        truncated = False

        if not self.car_inside():
            terminated = True

        self.lidar = self.track.get_distance_to_side(
            self.car.pos_x, self.car.pos_y, self.car.yaw, self.rays
        )
        observation = np.array([self.car.speed, *self.lidar.values()])

        # TODO call it manually?
        # if self.render_mode == "human":
        #     self.render(lidar)

        car_pos = Point(self.car.pos_x, self.car.pos_y)
        distance_to_centerline = self.track.center.distance(car_pos)
        road_side = self.track.center_polygon.contains(
            car_pos
        )  # to know which side of the road we're driving on
        distance_to_centerline = distance_to_centerline if road_side else -distance_to_centerline

        info = {"distance_to_centerline": distance_to_centerline}

        return observation, step_reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if self.random_init:
            x_min, y_min, x_max, y_max = self.track.outer.bounds

            while True:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                yaw = np.random.uniform(-np.pi, np.pi)
                init_state = np.array([x, y, yaw, 1e-9, 1e-9, 1e-9])

                self.car = Car(initial_state=init_state)

                if self.car_inside():
                    break
        else:
            init_state = np.array([1.0, 0.5, 1e-9, 1e-9, 1e-9, 1e-9])
            self.car = Car(initial_state=init_state)

        return self.step(np.array([0.0, 0.0]))[0], {}

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        # render the track
        for line in [self.track.inner, self.track.outer]:
            # Transform track points to screen points and scale
            points = [(SCALE * p[0], SCALE * p[1]) for p in line.coords]

            # Draw lines with pygame
            pygame.draw.lines(self.surf, (255, 255, 255), False, points, 2)

        global_vertices = self.car.vertices
        global_vertices = [(x * SCALE, y * SCALE) for x, y in global_vertices]

        # scale car position
        pos_x = self.car.pos_x * SCALE
        pos_y = self.car.pos_y * SCALE

        # render the lidar measurements as lines from the car position
        for angle, distance in self.lidar.items():
            # Calculate the end point of the line based on car's position, orientation, and lidar measurements
            end_x = self.car.pos_x * SCALE + distance * SCALE * np.cos(self.car.yaw + angle)
            end_y = self.car.pos_y * SCALE + distance * SCALE * np.sin(self.car.yaw + angle)

            # Draw the lidar line using pygame
            pygame.draw.line(self.surf, (0, 255, 0), (pos_x, pos_y), (end_x, end_y), 1)

        # Draw the car's rectangle on the screen using pygame
        pygame.draw.polygon(self.surf, (255, 255, 255), global_vertices)

        # pygame plumbing
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        assert self.screen is not None
        self.screen.fill(0)
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()

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
                    a[1] += 0.2
                if event.key == pygame.K_DOWN:
                    a[1] -= 0.2
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0

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
            env.render()
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
                print(f"obs: {s}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
