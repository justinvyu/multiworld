from typing import Any, Dict, Optional, Sequence

import os
import numpy as np
import collections
import matplotlib.pyplot as plt

from gym import spaces
import multiworld
from multiworld.core.multitask_env import MultitaskEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv
from serializable import Serializable


class PusherEnv(MujocoEnv, MultitaskEnv, Serializable):

    MODEL_PATH = os.path.join(
        os.path.dirname(multiworld.envs.__file__),
        'assets/pusher2d/pusher2d_simple.xml')

    def __init__(self,
                 init_qpos_range=None,
                 init_object_pos_range=None,
                 target_pos_range=None,
                 reset_gripper=True,
                 reset_object=True):

        self._Serializable__initialize(locals())
        self.reset_already = False
        self.goal_pos = np.array([0., 0.])
        MujocoEnv.__init__(self, model_path=self.MODEL_PATH, frame_skip=5)
        self.model.stat.extent = 10

        self._reset_gripper = reset_gripper
        self._reset_object = reset_object

        # === Initialize action space ===
        u = np.ones((3,)) / 2.
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        # === Initialize observation spaces ===
        boundary_dist = 2.0
        qpos_range = spaces.Box(
            low=np.array([-boundary_dist, -boundary_dist, -np.pi]),
            high=np.array([boundary_dist, boundary_dist, np.pi]),
            dtype=np.float32)
        qvel_range = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=qpos_range.shape,
            dtype=np.float32)
        box_range = spaces.Box(
            low=np.array([-boundary_dist, -boundary_dist]),
            high=np.array([boundary_dist, boundary_dist]),
            dtype=np.float32)
        object_vel_range = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=box_range.shape,
            dtype=np.float32)

        # self.observation_space = spaces.Dict([
        #     ('gripper_qpos', qpos_range),
        #     ('gripper_qvel', qvel_range),
        #     ('object_pos', box_range),
        #     ('object_vel', object_vel_range),
        #     ('target_pos', box_range),
        # ])

        # === Initialize reset ranges ===
        self._init_qpos_range = qpos_range
        self.set_init_qpos_range(init_qpos_range)

        self._init_object_pos_range = box_range
        self.set_init_object_pos_range(init_object_pos_range)

        self._target_pos_range = box_range
        self.set_target_pos_range(target_pos_range)

        self.reset()

    def set_init_qpos_range(self, init_qpos_range):
        if init_qpos_range:
            if isinstance(init_qpos_range, tuple):
                low, high = init_qpos_range
                self._init_qpos_range = spaces.Box(
                    np.array(low), np.array(high), dtype=np.float32)
            elif isinstance(init_qpos_range, list):
                self._init_qpos_range = np.array(init_qpos_range)
            elif isinstance(init_qpos_range, spaces.Box):
                self._init_qpos_range = init_qpos_range

    def set_init_object_pos_range(self, init_object_pos_range):
        if init_object_pos_range:
            if isinstance(init_object_pos_range, tuple):
                low, high = init_object_pos_range
                self._init_object_pos_range = spaces.Box(
                    np.array(low), np.array(high), dtype=np.float32)
            elif isinstance(init_object_pos_range, list):
                self._init_object_pos_range = np.array(init_object_pos_range)
            elif isinstance(init_object_pos_range, spaces.Box):
                self._init_object_pos_range = init_object_pos_range

    def set_target_pos_range(self, target_pos_range):
        if target_pos_range:
            if isinstance(target_pos_range, tuple):
                low, high = target_pos_range
                self._target_pos_range = spaces.Box(
                    np.array(low), np.array(high), dtype=np.float32)
            elif isinstance(target_pos_range, list):
                self._target_pos_range = np.array(target_pos_range)
            elif isinstance(target_pos_range, spaces.Box):
                self._target_pos_range = target_pos_range

    @property
    def target_pos_range(self):
        return self._target_pos_range

    def _sample_init_qpos(self):
        if isinstance(self._init_qpos_range, spaces.Box):
            return self._init_qpos_range.sample()
        elif isinstance(self._init_qpos_range, np.ndarray):
            # TODO: Different options for discrete # of init positions
            rand_idx = np.random.randint(len(self._init_qpos_range))
            return self._init_qpos_range[rand_idx]

    def _sample_init_object_pos(self):
        if isinstance(self._init_object_pos_range, spaces.Box):
            return self._init_object_pos_range.sample()
        elif isinstance(self._init_object_pos_range, np.ndarray):
            # TODO: Different options for discrete # of init positions
            rand_idx = np.random.randint(len(self._init_object_pos_range))
            return self._init_object_pos_range[rand_idx]

    def reset(self):
        self.goal_pos = self.sample_goal()['state_desired_goal']

        prev_obs = self._get_obs()
        if self._reset_gripper or self._reset_object or not self.reset_already:
            if self._reset_gripper or not self.reset_already:
                init_qpos = self._sample_init_qpos()
            else:
                init_qpos = prev_obs['gripper_qpos']

            if self._reset_object or not self.reset_already:
                object_pos = self._sample_init_object_pos()
            else:
                object_pos = prev_obs['object_pos']

            qpos = np.concatenate([init_qpos, object_pos])
            qvel = np.zeros(qpos.shape)
            self.set_state(np.array(qpos), np.array(qvel))
            self.reset_already = True

        target_id = self.model.body_name2id('target')
        bp = self.model.body_pos.copy()
        bp[target_id, :2] = self.goal_pos
        self.model.body_pos[:] = bp
        self.sim.forward()

        return self._get_obs()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        data = self.sim.data
        obs = collections.OrderedDict((
            ('gripper_qpos', np.array(data.qpos.flat[:3])),
            ('gripper_qvel', np.array(data.qvel.flat[:3])),
            ('object_pos', np.array(data.geom_xpos[-2:-1, :2].flat)),
            ('object_vel', np.array(data.qvel.flat[3:])),
            ('target_pos', self.goal_pos.copy()),
            ('state_achieved_goal', np.array(data.geom_xpos[-2:-1, :2].flat)),
            ('state_desired_goal', self.goal_pos.copy()),
        ))
        return obs

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs()

        curr_gripper_pos = self.sim.data.site_xpos[0, :2]
        curr_block_pos = next_obs['object_pos']

        dist_to_block = np.linalg.norm(curr_gripper_pos - curr_block_pos)
        block_dist = np.linalg.norm(self.goal_pos - curr_block_pos)
        reward = - dist_to_block - block_dist
        done = False
        return next_obs, reward, done, {
            'gripper_to_object_distance': dist_to_block,
            'object_to_target_distance': block_dist,
        }
        # self.do_simulation(action, self.frame_skip)
        # obs = self._get_obs()
        # reward = self.compute_reward(action, obs)
        # done = False
        # return obs, reward, done, {
        #     'gripper_to_object_distance': np.linalg.norm(
        #         obs['gripper_qpos'][:2] - obs['object_pos']),
        #     'object_to_target_distance': np.linalg.norm(
        #         obs['object_pos'] - obs['target_pos']),
        # }

    """
    MultitaskEnv interface
    """

    def compute_rewards(self, actions, obs):
        curr_gripper_pos = obs['gripper_qpos'][:, :2]
        curr_block_pos = obs['object_pos']
        target_block_pos = obs['target_pos']

        dist_to_block = np.linalg.norm(curr_gripper_pos - curr_block_pos, axis=-1)
        block_to_target = np.linalg.norm(target_block_pos - curr_block_pos, axis=-1)

        # TODO: Implement dist_to_block, block_to_target reward ratio
        reward = -(dist_to_block + block_to_target)
        return reward

    def get_goal(self):
        return {
            'desired_goal': self._target_pos.copy(),
            'state_desired_goal': self._target_pos.copy(),
        }

    def sample_goals(self, batch_size):
        if isinstance(self._target_pos_range, spaces.Box):
            goals = np.vstack([
                self._target_pos_range.sample()
                for _ in range(batch_size)
            ])
        elif isinstance(self._target_pos_range, np.ndarray):
            rand_idxs = np.random.randint(
                len(self._target_pos_range), size=batch_size)
            goals = self._target_pos_range[rand_idxs]
        return {'desired_goal': goals, 'state_desired_goal': goals}

    """
    Viewer setup and Rendering
    """

    def camera_init_fn(self):
        def init_fn(cam):
            cam.lookat[:3] = [0, 0, -0.28]
            cam.distance = 6.
            cam.elevation = -90
            cam.azimuth = 90
        return init_fn

    def viewer_setup(self):
        init_fn = self.camera_init_fn()
        init_fn(self.viewer.cam)

    # def render(self, mode='rgb_array', width=256, height=256):
    #     if mode == 'rgb_array':
    #         if not self.viewer:
    #             self.initialize_camera(self.camera_init_fn())
    #             self.viewer = True
    #         img = self.sim.render(width=width, height=height, mode='offscreen')
    #         return img
    #     elif mode == 'human':
    #         super(PusherEnv, self).render(mode=mode)
    #     else:
    #         raise NotImplementedError


env = PusherEnv()

# if __name__ == "__main__":
#     env = PusherEnv(
#         init_qpos_range=((-2.25, 0, 0), (-2.25, 0, 0)),
#         # init_qpos_range=((-1.5, -1.5), (1.5, 1.5)),
#         init_object_pos_range=[(1, 0)],
#         target_pos_range=((1.5, 1.5), (2, 2)),
#         # target_pos_range=[(2, 2), (2, -2)],
#         reset_gripper=True,
#         reset_object=True)

#     # act = 0
#     for _ in range(100):
#         env.reset()
#         for _ in range(50):
#             # env.step(np.array([act, 0, 0]))
#             obs, rew, done, info = env.step(env.action_space.sample())
#             print(rew, obs)
#             img = env.render(mode='human')
#         # print(act, env._get_obs()['gripper_qpos'])
#         # act += 0.1
#             # img = env.render(mode='rgb_array')
#             # plt.imshow(img)
#             # plt.show()
