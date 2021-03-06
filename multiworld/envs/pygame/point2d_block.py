from collections import OrderedDict
import logging

import numpy as np
from gym import spaces
from pygame import Color

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)
from multiworld.envs.pygame.pygame_viewer import PygameViewer
from multiworld.envs.pygame.walls import VerticalWall, HorizontalWall


class Point2DEnv(MultitaskEnv, Serializable):
    """
    A little 2D point whose life goal is to reach a target.
    """

    def __init__(
            self,
            render_dt_msec=0,
            action_l2norm_penalty=0,  # disabled for now
            render_onscreen=False,
            render_size=84,
            reward_type="dense",
            action_scale=1.0,
            target_radius=0.60,
            boundary_dist=4,
            ball_radius=0.50,
            walls=None,
            fixed_goal=None,
            fixed_reset=None,
            randomize_position_on_reset=True,
            images_are_rgb=False,  # else black and white
            show_goal=True,
            reset_free=False,
            **kwargs
    ):
        if walls is None:
            walls = []
        if walls is None:
            walls = []
        if fixed_goal is not None:
            fixed_goal = np.array(fixed_goal)
        if fixed_reset is not None:
            assert not randomize_position_on_reset
            self._fixed_position = fixed_reset
        if len(kwargs) > 0:
            LOGGER = logging.getLogger(__name__)
            LOGGER.log(logging.WARNING, "WARNING, ignoring kwargs:", kwargs)
        self.quick_init(locals())
        self.render_dt_msec = render_dt_msec
        self.action_l2norm_penalty = action_l2norm_penalty
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.reward_type = reward_type
        self.action_scale = action_scale
        self.target_radius = target_radius
        self.boundary_dist = boundary_dist
        self.ball_radius = ball_radius
        self.walls = walls
        self.fixed_goal = fixed_goal
        self.randomize_position_on_reset = randomize_position_on_reset
        self.images_are_rgb = images_are_rgb
        self.show_goal = show_goal
        self.ignore_previous_position = False
        self.reset_free = reset_free

        self.max_target_distance = self.boundary_dist - self.target_radius

        self._target_position = None
        self._position = np.zeros((2))

        u = np.ones(2)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        o = self.boundary_dist * np.ones(2)
        self.obs_range = spaces.Box(-o, o, dtype='float32')
        self.observation_space = spaces.Dict([
            ('observation', self.obs_range),
            ('desired_goal', self.obs_range),
            ('achieved_goal', self.obs_range),
            ('state_observation', self.obs_range),
            ('state_desired_goal', self.obs_range),
            ('state_achieved_goal', self.obs_range),
        ])

        self.drawer = None
        self.render_drawer = None
        self.reset_once = False

    def step(self, velocities):
        assert self.action_scale <= 1.0
        velocities = np.clip(velocities, a_min=-1, a_max=1) * self.action_scale
        new_position = self._position + velocities
        return self.step_by_position(new_position, velocities)

    def step_by_position(self, new_position, action):
        orig_new_pos = new_position.copy()
        for wall in self.walls:
            needs_collision_check = (
                not self.ignore_previous_position or
                wall.contains_point(new_position)
            )

            if needs_collision_check:
                new_position = wall.handle_collision(
                    self._position, new_position
                )
        if sum(new_position != orig_new_pos) > 1:
            """
            Hack: sometimes you get caught on two walls at a time. If you
            process the input in the other direction, you might only get
            caught on one wall instead.
            """
            new_position = orig_new_pos.copy()
            for wall in self.walls[::-1]:
                new_position = wall.handle_collision(
                    self._position, new_position
                )

        self._position = new_position
        self._position = np.clip(
            self._position,
            a_min=-self.boundary_dist,
            a_max=self.boundary_dist,
        )
        distance_to_target = np.linalg.norm(
            self._position - self._target_position
        )
        is_success = distance_to_target < self.target_radius

        ob = self._get_obs()

        reward = self.compute_reward(action, ob)
        info = {
            'radius': self.target_radius,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'is_success': is_success,
        }
        done = False
        return ob, reward, done, info


    def reset(self, goal=None):
        if self.reset_free is False or self.reset_once is False:
            if goal is not None:
                self._target_position = goal['state_desired_goal']
            else:
                self._target_position = self.sample_goal()['state_desired_goal']
            if self.randomize_position_on_reset:
                self._position = self._sample_position(
                    self.obs_range.low,
                    self.obs_range.high,
                )
            else:
                self._position = self._fixed_position
            self.reset_once = True
        return self._get_obs()

    def _position_inside_wall(self, pos):
        for wall in self.walls:
            if wall.contains_point(pos):
                return True
        return False

    def _sample_position(self, low, high):
        pos = np.random.uniform(low, high)
        while self._position_inside_wall(pos) is True:
            pos = np.random.uniform(low, high)
        return pos

    def _get_obs(self):
        return dict(
            observation=self._position.copy(),
            desired_goal=self._target_position.copy(),
            achieved_goal=self._position.copy(),
            state_observation=self._position.copy(),
            state_desired_goal=self._target_position.copy(),
            state_achieved_goal=self._position.copy(),
        )

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.target_radius).astype(np.float32)
        elif self.reward_type == "dense":
            return -d
        elif self.reward_type == 'vectorized_dense':
            return -np.abs(achieved_goals - desired_goals)
        else:
            raise NotImplementedError()

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'radius',
            'target_position',
            'distance_to_target',
            'velocity',
            'speed',
            'is_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    def get_goal(self):
        return {
            'desired_goal': self._target_position.copy(),
            'state_desired_goal': self._target_position.copy(),
        }

    def sample_goals(self, batch_size):
        if not self.fixed_goal is None:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.zeros((batch_size, self.obs_range.low.size))
            for b in range(batch_size):
                if batch_size > 1:
                    logging.warning("This is very slow!")
                goals[b, :] = self._sample_position(
                    self.obs_range.low,
                    self.obs_range.high,
                )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_position(self, pos):
        self._position[0] = pos[0]
        self._position[1] = pos[1]

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None):
        """Returns a black and white image"""
        if self.drawer is None:
            if width != height:
                raise NotImplementedError()
            self.drawer = PygameViewer(
                screen_width=width,
                screen_height=height,
                x_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                y_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                render_onscreen=self.render_onscreen,
            )
        self.draw(self.drawer, False)
        img = self.drawer.get_image()
        if self.images_are_rgb:
            return img.transpose((1, 0, 2))
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = (-r + b).transpose().flatten()
            return img

    def set_to_goal(self, goal_dict):
        goal = goal_dict["state_desired_goal"]
        self._position = goal
        self._target_position = goal

    def get_env_state(self):
        return self._get_obs()

    def set_env_state(self, state):
        position = state["state_observation"]
        goal = state["state_desired_goal"]
        self._position = position
        self._target_position = goal

    def draw(self, drawer, tick):
        # if self.drawer is not None:
        #     self.drawer.fill(Color('white'))
        # if self.render_drawer is not None:
        #     self.render_drawer.fill(Color('white'))
        drawer.fill(Color('white'))
        if self.show_goal:
            drawer.draw_solid_circle(
                self._target_position,
                self.target_radius,
                Color('green'),
            )
        drawer.draw_solid_circle(
            self._position,
            self.ball_radius,
            Color('blue'),
        )

        for wall in self.walls:
            drawer.draw_segment(
                wall.endpoint1,
                wall.endpoint2,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint2,
                wall.endpoint3,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint3,
                wall.endpoint4,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint4,
                wall.endpoint1,
                Color('black'),
            )

        drawer.render()
        if tick:
            drawer.tick(self.render_dt_msec)

    def render(self, mode, width=50, height=50):
        return self.get_image(width=50, height=50)

#    def render(self, mode=, close=False):
#        if close:
#            self.render_drawer = None
#            return
#
#        if self.render_drawer is None or self.render_drawer.terminated:
#            self.render_drawer = PygameViewer(
#                self.render_size,
#                self.render_size,
#                x_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
#                y_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
#                render_onscreen=True,
#            )
#        self.draw(self.render_drawer, True)

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'distance_to_target',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    """Static visualization/utility methods"""

    @staticmethod
    def true_model(state, action):
        velocities = np.clip(action, a_min=-1, a_max=1)
        position = state
        new_position = position + velocities
        return np.clip(
            new_position,
            a_min=-Point2DEnv.boundary_dist,
            a_max=Point2DEnv.boundary_dist,
        )

    @staticmethod
    def true_states(state, actions):
        real_states = [state]
        for action in actions:
            next_state = Point2DEnv.true_model(state, action)
            real_states.append(next_state)
            state = next_state
        return real_states

    @staticmethod
    def plot_trajectory(ax, states, actions, goal=None):
        assert len(states) == len(actions) + 1
        x = states[:, 0]
        y = -states[:, 1]
        num_states = len(states)
        plasma_cm = plt.get_cmap('plasma')
        for i, state in enumerate(states):
            color = plasma_cm(float(i) / num_states)
            ax.plot(state[0], -state[1],
                    marker='o', color=color, markersize=10,
                    )

        actions_x = actions[:, 0]
        actions_y = -actions[:, 1]

        ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
                  scale_units='xy', angles='xy', scale=1, width=0.005)
        ax.quiver(x[:-1], y[:-1], actions_x, actions_y, scale_units='xy',
                  angles='xy', scale=1, color='r',
                  width=0.0035, )
        ax.plot(
            [
                -Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                -Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )

        if goal is not None:
            ax.plot(goal[0], -goal[1], marker='*', color='g', markersize=15)
        ax.set_ylim(
            -Point2DEnv.boundary_dist - 1,
            Point2DEnv.boundary_dist + 1,
        )
        ax.set_xlim(
            -Point2DEnv.boundary_dist - 1,
            Point2DEnv.boundary_dist + 1,
        )

    def initialize_camera(self, init_fctn):
        pass


class Point2DWallEnv(Point2DEnv):
    """Point2D with walls"""

    def __init__(
            self,
            wall_shape="",
            wall_thickness=1.0,
            inner_wall_max_dist=1,
            **kwargs
    ):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.inner_wall_max_dist = inner_wall_max_dist
        self.wall_shape = wall_shape
        self.wall_thickness = wall_thickness
        if wall_shape == "u":
            self.walls = [
                # Right wall
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ]
        if wall_shape == "-" or wall_shape == "h":
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ]
        if wall_shape == "--":
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ]
        if wall_shape == "big-u":
            self.walls = [
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                    # -self.inner_wall_max_dist,
                    # self.inner_wall_max_dist,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "easy-u":
            self.walls = [
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*0.5,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*0.5,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "big-h":
            self.walls = [
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                ),
            ]
        if wall_shape == "box":
            self.walls = [
                # Bottom wall
                VerticalWall(
                    self.ball_radius,
                    0,
                    0,
                    0,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "none":
            self.walls = []


class Point2DBlockEnv(Point2DEnv):
    def __init__(self, block_matrix, action_as_position=False,
                 random_steps=1, random_step_variance=.2,
                 *args, **kwargs):
        self.quick_init(locals())
        super().__init__(*args, **kwargs)
        self.walls = []
        self.scale_factor = self.boundary_dist * 2 / len(block_matrix)
        self.inner_wall_max_dist = 1
        self.wall_thickness = 0.5
        self.block_matrix = block_matrix
        self.action_as_position = action_as_position
        self.random_steps = random_steps
        self.random_step_variance = random_step_variance
        self.ignore_previous_position = True

        self.num_empty_blocks = 0

        self.coverage_map = [
            [False for _ in row]
            for row in self.block_matrix
        ]

        for row_idx, row in enumerate(block_matrix):
            for col_idx, val in enumerate(row):
                if val:
                    # Because the window is (-boundary_dist, boundary_dist)
                    # instead of (0, 2 * boundary_dist)
                    # y = -self.boundary_dist + row_idx *block_width
                    # x = -self.boundary_dist + col_idx * block_width
                    x, y = self.get_block_xy(row_idx, col_idx)
                    self.walls.append(
                        VerticalWall(
                            0,
                            x_pos = x + self.block_width / 2,
                            bottom_y = y + self.block_width / 2,
                            top_y = y + self.block_width / 2,
                            thickness = self.block_width / 2
                        ),
                    )
                else:
                    self.num_empty_blocks += 1
        self.mode = "train"

    @property
    def block_width(self):
        return self.scale_factor

    @property
    def coverage(self):
        num_blocks_explored = 0
        for row in self.coverage_map:
            for is_explored in row:
                if is_explored:
                    num_blocks_explored += 1
        return num_blocks_explored / self.num_empty_blocks

    def get_block_xy(self, block_row_idx, block_col_idx):
        y = -self.boundary_dist + block_row_idx * self.block_width
        x = -self.boundary_dist + block_col_idx * self.block_width
        return (x, y)

    def xy_to_block_idx(self, x, y):
        block_col = (x + self.boundary_dist) // self.block_width
        block_row = (y + self.boundary_dist) // self.block_width
        return int(block_row), int(block_col)

    def debug_info(self):
        info = OrderedDict()
        info['env/coverage'] = self.coverage
        return info

    def get_eval_paths(self):
        goals = []
        for row_idx, row in enumerate(block_matrix):
            for col_idx, val in enumerate(row):
                # This is an open spot. We assume it is reachable from the start
                # position.
                if not val:
                    goals.append(get_block_xy(row_idx, col_idx))
        return goals

    def train(self):
        self.mode = "train"

    def test(self):
        self.mode = "test"

    def handle_coverage(self, obs):
        xy = obs['state_achieved_goal']
        block_row, block_col = self.xy_to_block_idx(xy[0], xy[1])
        if (
            block_row >= len(self.coverage_map) or
            block_col >= len(self.coverage_map[0])
        ):
            return
        if self.block_matrix[block_row][block_col]:
            return
        self.coverage_map[block_row][block_col] = True

    def step(self, action):
        if self.action_as_position:
            action = np.clip(action, -self.boundary_dist, self.boundary_dist)
            ob, reward, done, info = self.step_by_position(action, action)
            for _ in range(self.random_steps):
                random_action = np.random.normal(0, self.random_step_variance, 2)
                self.ignore_previous_position = False
                ob, reward, done, info = super().step(random_action)
                self.ignore_previous_position = True
            self.handle_coverage(ob)
            return ob, reward, done, info
        else:
            obs, reward, done, info = super().step(action)
            self.handle_coverage(obs)
            return obs, reward, done, info

class OffsetEnv(Point2DBlockEnv):
    def __init__(self, offset_x, offset_y, *args, **kwargs):
        self.quick_init(locals())
        super().__init__(*args, **kwargs)
        self.offset_x = offset_x
        self.offset_y = offset_y

    def offset_obs(self, obs):
        new_obs = {}
        for k, v in obs.items():
            x, y = v[0], v[1]
            new_obs[k] = np.array([x + self.offset_x, y + self.offset_y])
        return new_obs

    def unoffset_action(self, action):
        action_x, action_y = action[0], action[1]
        return np.array([action_x - self.offset_x, action_y - self.offset_y])

    def reset(self):
        obs = super().reset()
        obs = self.offset_obs(obs)
        return obs

    def step(self, action):
        action = self.unoffset_action(action)
        obs, reward, done, info = super().step(action)
        obs = self.offset_obs(obs)
        return obs, reward, done, info


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    # e = gym.make('Point2D-Box-Wall-v1')
    # e = gym.make('Point2D-Big-UWall-v1')
    # e = gym.make('Point2D-Easy-UWall-v1')
    for i in range(1000):
        e.reset()
        for j in range(5):
            e.step(np.random.rand(2))
            e.render()
            im = e.get_image()