from collections import OrderedDict
import logging

import numpy as np
from gym import spaces
from pygame import Color
import collections

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
            render_size=256,
            reward_type="dense",
            action_scale=1.0,
            target_radius=0.1,
            boundary_dist=4,
            ball_radius=0.15,
            walls=None,
            init_pos_range=None,
            target_pos_range=None,
            images_are_rgb=False,  # else black and white
            show_goal=True,
            n_bins=32,
            use_count_reward=False,
            show_discrete_grid=False,
            fix_goal_position=False,
            multiple_goals=False,
            goal_position=None,
            **kwargs
    ):
        if walls is None:
            walls = []
        if walls is None:
            walls = []
        # if fixed_goal is not None:
        #     fixed_goal = np.array(fixed_goal)

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
        self.n_bins = n_bins
        self.use_count_reward = use_count_reward
        self.show_discrete_grid = show_discrete_grid
        self.images_are_rgb = images_are_rgb
        self.show_goal = show_goal

        self.x_bins = np.linspace(-self.boundary_dist, self.boundary_dist, self.n_bins)
        self.y_bins = np.linspace(-self.boundary_dist, self.boundary_dist, self.n_bins)
        self.bin_counts = np.ones((self.n_bins + 1, self.n_bins + 1))

        self.max_target_distance = self.boundary_dist - self.target_radius

        self._target_position = None
        self._position = np.zeros((2))

        u = np.ones(2)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        o = self.boundary_dist * np.ones(2)
        self.obs_range = spaces.Box(-o, o, dtype='float32')

        if not init_pos_range:
            self.init_pos_range = self.obs_range
        else:
            assert np.all(np.abs(init_pos_range) <= boundary_dist), ("Init position must be"
                f"within the boundaries of the environment: ({-boundary_dist}, {boundary_dist})")
            low, high = init_pos_range
            self.init_pos_range = spaces.Box(
                np.array(low), np.array(high), dtype='float32')

        if not target_pos_range:
            self.target_pos_range = self.obs_range
        else:
            assert np.all(np.abs(target_pos_range) <= boundary_dist), ("Goal position must be"
                f"within the boundaries of the environment: ({-boundary_dist}, {boundary_dist})")

            low, high = target_pos_range
            self.target_pos_range = spaces.Box(
                np.array(low), np.array(high), dtype='float32')

        self.observation_space = spaces.Dict([
            ('observation', self.obs_range),
            ('onehot_observation', spaces.Box(
                0, 1, shape=(2 * (self.n_bins + 1), ), dtype=np.float32)),
            ('desired_goal', self.obs_range),
            ('achieved_goal', self.obs_range),
            ('state_observation', self.obs_range),
            ('state_desired_goal', self.obs_range),
            ('state_achieved_goal', self.obs_range),
        ])

        self.drawer = None
        self.render_drawer = None

        self.fix_goal_position = fix_goal_position
        self.multiple_goals = multiple_goals
        if goal_position is not None and fix_goal_position:
            self.goal_position = np.array(goal_position, dtype='float32')

        self.reset()

    def step(self, velocities):
        assert self.action_scale <= 1.0
        velocities = np.clip(velocities, a_min=-1, a_max=1) * self.action_scale
        new_position = self._position + velocities
        orig_new_pos = new_position.copy()
        for wall in self.walls:
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

        if self.multiple_goals:
            distance_to_target = np.min([np.linalg.norm(self._position - t) \
                for t in self._target_position])
        else:
            distance_to_target = np.linalg.norm(
                self._position - self._target_position
            )
        is_success = distance_to_target < self.target_radius

        ob = self._get_obs()
        x_d, y_d = ob['discrete_observation']
        self.bin_counts[x_d, y_d] += 1

        reward = self.compute_reward(velocities, ob)
        info = {
            'radius': self.target_radius,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
            'is_success': is_success,
        }

        done = False
        return ob, reward, done, info

    def reset(self):
        # TODO: Make this more general
        if self.fix_goal_position:
            self._target_position = self.goal_position
        else:
            self._target_position = self.sample_goal()['state_desired_goal']

        # if self.randomize_position_on_reset:
        self._position = self._sample_position(
            # self.obs_range.low,
            # self.obs_range.high,
            self.init_pos_range.low,
            self.init_pos_range.high,
        )
        ob = self._get_obs()
        x_d, y_d = ob['discrete_observation']
        self.bin_counts[x_d, y_d] += 1

        return self._get_obs()

    def _position_inside_wall(self, pos):
        for wall in self.walls:
            if wall.contains_point(pos):
                return True
        return False

    def _sample_position(self, low, high):
        if np.all(low == high):
            return low
        pos = np.random.uniform(low, high)
        while self._position_inside_wall(pos) is True:
            pos = np.random.uniform(low, high)
        return pos

    def clear_bin_counts(self):
        self.bin_counts = np.ones((self.n_bins + 1, self.n_bins + 1))

    def _discretize_observation(self, obs):
        if isinstance(obs, dict):
            x, y = obs['state_observation'].copy()
            x_d, y_d = np.digitize(x, self.x_bins), np.digitize(y, self.y_bins)
            return np.array([x_d, y_d])
        else:
            assert isinstance(obs, np.ndarray) and obs.ndim == 2 and obs.shape[1] == 2
            x_d = np.expand_dims(np.digitize(obs[:, 0], self.x_bins), 1)
            y_d = np.expand_dims(np.digitize(obs[:, 1], self.y_bins), 1)
            return np.concatenate((x_d, y_d), axis=1)

    def get_bin_counts(self, obs):
        obs_d = self._discretize_observation(obs)
        return self.bin_counts[obs_d[:, 0], obs_d[:, 1]]

    def get_count_bonuses(self, obs):
        obs_d = self._discretize_observation(obs)

        # TODO: give multiple options for count bonus
        return 1 / np.sqrt(self.bin_counts[obs_d[:, 0], obs_d[:, 1]])

    def _get_obs(self):
        obs = collections.OrderedDict(
            observation=self._position.copy(),
            desired_goal=self._target_position.copy(),
            achieved_goal=self._position.copy(),
            state_observation=self._position.copy(),
            state_desired_goal=self._target_position.copy(),
            state_achieved_goal=self._position.copy(),
            image_observation=self.render(mode='rgb_array')
        )

        # Update with discretized state
        pos_discrete = self._discretize_observation(obs)
        pos_onehot = np.eye(self.n_bins + 1)[pos_discrete]
        obs['discrete_observation'] = pos_discrete
        obs['onehot_observation'] = pos_onehot

        return obs

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']

        if self.multiple_goals:
            d = np.min(np.linalg.norm(np.repeat(achieved_goals, \
                desired_goals.shape[-2], axis=-2)[None] - desired_goals, axis=-1),
                axis=-1)
        else:
            d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)

        if self.reward_type == "sparse":
            r = -(d > self.target_radius).astype(np.float32)
        elif self.reward_type == "sparse-positive":
            r = (d < self.target_radius).astype(np.float32)
        elif self.reward_type == "dense":
            r = -d
        elif self.reward_type == "vectorized_dense":
            r = -np.abs(achieved_goals - desired_goals)
        elif self.reward_type == "none":
            r = np.zeros(d.shape)
        else:
            raise NotImplementedError()

        if self.use_count_reward:
            # TODO: Add different count based strategies
            pos_d = obs['discrete_observation']
            r += 1 / np.sqrt(self.bin_counts[pos_d[:, 0], pos_d[:, 1]])

        return r

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
        # if self.fixed_goal:
        #     goals = np.repeat(
        #         self.fixed_goal.copy()[None],
        #         batch_size,
        #         0)
        # else:
        goals = np.zeros((batch_size, self.obs_range.low.size))
        for b in range(batch_size):
            if batch_size > 1:
                logging.warning("This is very slow!")
            goals[b, :] = self._sample_position(
                # self.obs_range.low,
                # self.obs_range.high,
                self.target_pos_range.low,
                self.target_pos_range.high,
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_position(self, pos):
        self._position[0] = pos[0]
        self._position[1] = pos[1]

        ob = self._get_obs()
        x_d, y_d = ob['discrete_observation']
        self.bin_counts[x_d, y_d] += 1

    def set_goal(self, goal):
        self._target_position = goal

    def get_target_position(self):
        return self._target_position

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None):
        """Returns a black and white image"""
        if self.drawer is None:
            if width != height:
                raise NotImplementedError()
            self.drawer = PygameViewer(
                screen_width=width,
                screen_height=height,
                # TODO(justinvyu): Action scale = 1 breaks rendering, why?
                # x_bounds=(-self.boundary_dist - self.ball_radius,
                #           self.boundary_dist + self.ball_radius),
                # y_bounds=(-self.boundary_dist - self.ball_radius,
                #           self.boundary_dist + self.ball_radius),
                x_bounds=(-self.boundary_dist,
                          self.boundary_dist),
                y_bounds=(-self.boundary_dist,
                          self.boundary_dist),
                render_onscreen=self.render_onscreen,
            )
        self.draw(self.drawer)
        img = self.drawer.get_image()
        if self.images_are_rgb:
            return img.transpose((1, 0, 2))
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = (-r + b).transpose().flatten()
            return img

    def set_to_goal(self, goal_dict):
        goal = goal_dict["state_desired_goal"]
        if self.multiple_goals:
            self._target_position = goal
            self._position = goal[np.random.choice(len(goal))]
        else:
            self._position = goal
            self._target_position = goal

    def get_env_state(self):
        return self._get_obs()

    def set_env_state(self, state):
        position = state["state_observation"]
        self.set_to_goal(state)
        self._position = position

    def draw(self, drawer):
        drawer.fill(Color('white'))

        if self.show_discrete_grid:
            for x in self.x_bins:
                drawer.draw_segment(
                    (x, -self.boundary_dist),
                    (x, self.boundary_dist),
                    Color(220,220,220,25), aa=False)
            for y in self.y_bins:
                drawer.draw_segment(
                    (-self.boundary_dist, y),
                    (self.boundary_dist, y),
                    Color(220,220,220,25), aa=False)

        if self.show_goal:
            if self.multiple_goals:
                for t in self._target_position:
                    drawer.draw_solid_circle(
                        t,
                        self.target_radius,
                        Color('green'),
                    )
            else:
                drawer.draw_solid_circle(
                    self._target_position,
                    self.target_radius,
                    Color('green'),
                )
        try:
            drawer.draw_solid_circle(
                self._position,
                self.ball_radius,
                Color('blue'),
            )
        except ValueError as e:
            print('\n\n RENDER ERROR \n\n')

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

    def render(self, mode='human', width=None, height=None, close=False):
        if close:
            self.render_drawer = None
            return
        if mode =='rgb_array':
            if width is None:
                width = self.render_size
            if height is None:
                height = self.render_size
            return self.get_image(width=width, height=height)

        if self.render_drawer is None or self.render_drawer.terminated:
            self.render_drawer = PygameViewer(
                self.render_size,
                self.render_size,
                # x_bounds=(-self.boundary_dist-self.ball_radius,
                #           self.boundary_dist+self.ball_radius),
                # y_bounds=(-self.boundary_dist-self.ball_radius,
                #           self.boundary_dist+self.ball_radius),
                x_bounds=(-self.boundary_dist,
                          self.boundary_dist),
                y_bounds=(-self.boundary_dist,
                          self.boundary_dist),
                render_onscreen=True,
            )
        self.draw(self.render_drawer)
        self.render_drawer.tick(self.render_dt_msec)
        if mode != 'interactive':
            self.render_drawer.check_for_exit()

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in ('distance_to_target', ):
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
            wall_shape="hard-maze",
            wall_thickness=0,
            inner_wall_max_dist=1,
            **kwargs
    ):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.inner_wall_max_dist = inner_wall_max_dist
        self.wall_shape = wall_shape
        self.wall_thickness = wall_thickness

        WALL_FORMATIONS = {
            "u": [
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
            ],
            "-": [
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ],
            "--": [
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ],
            "big-u": [
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
                    self.wall_thickness
                ),
            ],
            "easy-u": [
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
            ],
            "big-h": [
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                ),
            ],
            "box": [
                # Bottom wall
                VerticalWall(
                    self.ball_radius,
                    0,
                    0,
                    0,
                    self.wall_thickness
                ),
            ],
            "easy-maze": [
                VerticalWall(
                    self.ball_radius,
                    0,
                    -self.boundary_dist,
                    self.inner_wall_max_dist,
                ),
            ],
            "medium-maze": [
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist/3,
                    -self.boundary_dist,
                    self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    self.boundary_dist/3,
                    -self.inner_wall_max_dist,
                    self.boundary_dist
                ),
            ],
            "hard-maze": [
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    -self.boundary_dist,
                    self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    self.boundary_dist - self.inner_wall_max_dist,
                ),
                HorizontalWall(
                    self.ball_radius,
                    self.boundary_dist - self.inner_wall_max_dist,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    -self.boundary_dist + self.inner_wall_max_dist * 2,
                    self.boundary_dist - self.inner_wall_max_dist,
                ),
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist + self.inner_wall_max_dist * 2,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    0,
                ),
            ],
            None: [],
        }

        if wall_shape == "double-maze":
            WALL_FORMATIONS["double-maze"] = [
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist / 1.5,
                    -self.boundary_dist,
                    self.boundary_dist,
                ),
                HorizontalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist / 2,
                    -self.boundary_dist / 8,
                    self.boundary_dist / 8  + self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist / 8,
                    -self.boundary_dist / 1.5,
                    -self.inner_wall_max_dist / 2,
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist / 8 - self.inner_wall_max_dist,
                    -self.boundary_dist / 1.5 + self.inner_wall_max_dist,
                    self.boundary_dist / 1.5,
                ),
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist / 1.5 + self.inner_wall_max_dist,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    -self.boundary_dist / 8 - self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    -self.boundary_dist / 1.5 + self.inner_wall_max_dist,
                    self.boundary_dist / 1.5 - self.inner_wall_max_dist,
                ),
                HorizontalWall(
                    self.ball_radius,
                    self.boundary_dist / 1.5 - self.inner_wall_max_dist,
                    -self.boundary_dist + self.inner_wall_max_dist,
                    -self.boundary_dist / 8 - 2 * self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist / 8 - 2 * self.inner_wall_max_dist,
                    -self.boundary_dist / 1.5 + 2 * self.inner_wall_max_dist,
                    self.boundary_dist / 1.5 - self.inner_wall_max_dist,
                ),
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist / 1.5 + 2 * self.inner_wall_max_dist,
                    -self.boundary_dist + 2 * self.inner_wall_max_dist,
                    -self.boundary_dist / 8 - 2 * self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist + 2 * self.inner_wall_max_dist,
                    -self.boundary_dist / 1.5 + 2 * self.inner_wall_max_dist,
                    self.boundary_dist / 1.5 - 2 * self.inner_wall_max_dist,
                ),
                HorizontalWall(
                    self.ball_radius,
                    self.boundary_dist / 1.5 - 2 * self.inner_wall_max_dist,
                    -self.boundary_dist + 2 * self.inner_wall_max_dist,
                    -self.boundary_dist / 8 - 3 * self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist / 8 - 3 * self.inner_wall_max_dist,
                    -self.boundary_dist / 1.5 + 3 * self.inner_wall_max_dist,
                    self.boundary_dist / 1.5 - 2 * self.inner_wall_max_dist,
                ),
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist / 1.5 + 3 * self.inner_wall_max_dist,
                    -self.boundary_dist + 3 * self.inner_wall_max_dist,
                    -self.boundary_dist / 8 - 3 * self.inner_wall_max_dist,
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist + 3 * self.inner_wall_max_dist,
                    -self.boundary_dist / 1.5 + 3 * self.inner_wall_max_dist,
                    self.boundary_dist / 1.5 - 3 * self.inner_wall_max_dist,
                ),
            ]
            mirror_walls = []
            for wall in WALL_FORMATIONS['double-maze']:
                if isinstance(wall, HorizontalWall):
                    mirror_walls.append(HorizontalWall(
                        self.ball_radius,
                        -wall.min_y - wall.min_dist - wall.thickness,
                        -wall.max_x + wall.min_dist + wall.thickness,
                        -wall.min_x - wall.min_dist - wall.thickness,
                    ))
                elif isinstance(wall, VerticalWall):
                    mirror_walls.append(VerticalWall(
                        self.ball_radius,
                        -wall.min_x - wall.min_dist - wall.thickness,
                        -wall.max_y + wall.min_dist + wall.thickness,
                        -wall.min_y - wall.min_dist - wall.thickness,
                    ))
            WALL_FORMATIONS['double-maze'].extend(mirror_walls)

        if wall_shape == "rooms":
            room_width = 2 * self.boundary_dist / 3
            door_width = room_width / 3
            nondoor_width = (room_width - door_width) / 2

            y0 = x1 = self.boundary_dist - room_width
            y1 = x0 = -self.boundary_dist + room_width

            WALL_FORMATIONS["rooms"] = [
                # Top rooms
                HorizontalWall(self.ball_radius, y0, -self.boundary_dist, x0, wall_thickness),
                VerticalWall(self.ball_radius, x0, self.boundary_dist - nondoor_width, self.boundary_dist, wall_thickness),
                VerticalWall(self.ball_radius, x0, y0, y0 + nondoor_width, wall_thickness),

                HorizontalWall(self.ball_radius, y0, x1, self.boundary_dist, wall_thickness),
                VerticalWall(self.ball_radius, x1, self.boundary_dist - nondoor_width, self.boundary_dist, wall_thickness),
                VerticalWall(self.ball_radius, x1, y0, y0 + nondoor_width, wall_thickness),

                HorizontalWall(self.ball_radius, y0, x0, x0 + nondoor_width, wall_thickness),
                HorizontalWall(self.ball_radius, y0, x1 - nondoor_width, x1, wall_thickness),

                HorizontalWall(self.ball_radius, y1, -self.boundary_dist, x0, wall_thickness),
                HorizontalWall(self.ball_radius, y1, x1, self.boundary_dist, wall_thickness),

                VerticalWall(self.ball_radius, x0, y0 - nondoor_width, y0, wall_thickness),
                VerticalWall(self.ball_radius, x0, y1, y1 + nondoor_width, wall_thickness),
                VerticalWall(self.ball_radius, x1, y0 - nondoor_width, y0, wall_thickness),
                VerticalWall(self.ball_radius, x1, y1, y1 + nondoor_width, wall_thickness),

                HorizontalWall(self.ball_radius, y1, x0, x0 + nondoor_width, wall_thickness),
                HorizontalWall(self.ball_radius, y1, x1 - nondoor_width, x1, wall_thickness),

                VerticalWall(self.ball_radius, x0, y1 - nondoor_width, y1, wall_thickness),
                VerticalWall(self.ball_radius, x0, -self.boundary_dist, -self.boundary_dist + nondoor_width, wall_thickness),
                VerticalWall(self.ball_radius, x1, y1 - nondoor_width, y1, wall_thickness),
                VerticalWall(self.ball_radius, x1, -self.boundary_dist, -self.boundary_dist + nondoor_width, wall_thickness),
            ]
        if wall_shape == "rooms_large":
            room_width = 2 * self.boundary_dist / 3 / 2
            door_width = 1

            y0 = x1 = self.boundary_dist - room_width
            y1 = x0 = -self.boundary_dist + room_width

            WALL_FORMATIONS["rooms_large"] = []
            for j in range(5):
                for i in range(6):
                    WALL_FORMATIONS["rooms_large"].append(VerticalWall(self.ball_radius, x0 + room_width * j, \
                        self.boundary_dist - room_width * (i + 0.5) + door_width / 2, \
                        self.boundary_dist - room_width * i, wall_thickness))
                    WALL_FORMATIONS["rooms_large"].append(VerticalWall(self.ball_radius, x0 + room_width * j, \
                        self.boundary_dist - room_width * (i + 1), \
                        self.boundary_dist - room_width * (i + 0.5) - door_width/2, wall_thickness))

            for j in range(5):
                for i in range(6):
                    WALL_FORMATIONS["rooms_large"].append(HorizontalWall(self.ball_radius, y1 + j * room_width,\
                     self.boundary_dist - room_width * (i + 0.5) + door_width / 2, \
                     self.boundary_dist - room_width * i, wall_thickness))
                    WALL_FORMATIONS["rooms_large"].append(HorizontalWall(self.ball_radius, y1 + j * room_width,\
                     self.boundary_dist - room_width * (i + 1), \
                     self.boundary_dist - room_width * (i + 0.5) - door_width/2, wall_thickness))

            vertical_close = [(1, 2), (0, 3), (3, 5), (3, 3), (0, 1), (4, 5)]
            horizontal_close = [(1, 2), (3, 0), (4, 2), (0, 3), (2, 2), (4, 5), (2, 5)]
            for j, i in vertical_close:
                WALL_FORMATIONS["rooms_large"].append(VerticalWall(self.ball_radius, x0 + room_width * j, \
                    self.boundary_dist - room_width * (i + 0.5) - door_width / 2, \
                    self.boundary_dist - room_width * (i + 0.5) + door_width/2, wall_thickness))
            for j, i in horizontal_close:
                WALL_FORMATIONS["rooms_large"].append(HorizontalWall(self.ball_radius, x0 + room_width * j, \
                    self.boundary_dist - room_width * (i + 0.5) - door_width / 2, \
                    self.boundary_dist - room_width * (i + 0.5) + door_width/2, wall_thickness))



        self.walls = WALL_FORMATIONS.get(wall_shape, [])

if __name__ == "__main__":
    import gym
    import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import multiworld
    multiworld.register_all_envs()

    e = gym.make('Point2DFixed-v0', **{'reward_type': 'none', 'use_count_reward': True})
    # e = gym.make('Point2DSingleWall-v0')

    # e = gym.make('Point2D-Box-Wall-v1')
    # e = gym.make('Point2D-Big-UWall-v1')
    # e = gym.make('Point2D-Easy-UWall-v1')
    # e = gym.make('Point2DEnv-Image-v0')

    for i in range(100):
        e.reset()
        for j in range(100):
            obs, rew, done, info = e.step(e.action_space.sample())
            # e.render()
            # img = e.get_image()
            # plt.imshow(img)
            # plt.show()
            # print(rew)
            print(e.observation_space, obs['onehot_observation'])
