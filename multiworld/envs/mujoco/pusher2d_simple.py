from typing import Any, Dict, Optional, Sequence

import os
import numpy as np
import collections
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

from gym import spaces
import multiworld
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
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
                 reset_object=True,
                 do_reset=True,
                 multi_reset=False):
        self.do_reset = do_reset
        self.multi_reset = multi_reset

        self._Serializable__initialize(locals())
        self.reset_already = False
        self.goal_pos = np.array([2.0, 0.])
        MujocoEnv.__init__(self, model_path=self.MODEL_PATH, frame_skip=5)
        self.model.stat.extent = 10

        self._reset_gripper = reset_gripper
        self._reset_object = reset_object

        # === Initialize action space ===
        u = np.ones(3)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        # === Initialize observation spaces ===
        boundary_dist = 2.0
        self._qpos_range = spaces.Box(
            low=np.array([-boundary_dist, -boundary_dist, -np.pi]),
            high=np.array([boundary_dist, boundary_dist, np.pi]),
            dtype=np.float32)
        self._qvel_range = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._qpos_range.shape,
            dtype=np.float32)
        self._box_range = spaces.Box(
            low=np.array([-boundary_dist, -boundary_dist]),
            high=np.array([boundary_dist, boundary_dist]),
            dtype=np.float32)
        self.observation_space = spaces.Dict([
            ('gripper_qpos', self._qpos_range),
            ('gripper_qvel', self._qvel_range),
            ('object_pos', self._box_range),
            ('target_pos', self._box_range),
        ])

        # === Initialize reset ranges ===
        self._init_qpos_range = self._qpos_range
        if init_qpos_range:
            if isinstance(init_qpos_range, tuple):
                low, high = init_qpos_range
                self._init_qpos_range = spaces.Box(
                    np.array(low), np.array(high), dtype=np.float32)
            elif isinstance(init_qpos_range, list):
                self._init_qpos_range = np.array(init_qpos_range)

        self._init_object_pos_range = self._box_range
        if init_object_pos_range:
            if isinstance(init_object_pos_range, tuple):
                low, high = init_object_pos_range
                self._init_object_pos_range = spaces.Box(
                    np.array(low), np.array(high), dtype=np.float32)
            elif isinstance(init_object_pos_range, list):
                self._init_object_pos_range = np.array(init_object_pos_range)

        self._target_pos_range = self._box_range
        if target_pos_range:
            if isinstance(target_pos_range, tuple):
                low, high = target_pos_range
                self._target_pos_range = spaces.Box(
                    np.array(low), np.array(high), dtype=np.float32)
            elif isinstance(target_pos_range, list):
                self._target_pos_range = np.array(target_pos_range)

        self.reset()

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
        # TODO: Restrict the hinge angle to -np.pi, np.pi
        gripper_qpos = np.array([
            data.get_joint_qpos('wrist_slidex'), # x coord
            data.get_joint_qpos('wrist_slidey'), # y coord
            data.get_joint_qpos('wrist_hinge')   # angle
        ])
        gripper_qvel = np.array([
            data.get_joint_qvel('wrist_slidex'), # x coord
            data.get_joint_qvel('wrist_slidey'), # y coord
            data.get_joint_qvel('wrist_hinge')   # angle
        ])
        object_pos = data.get_geom_xpos('object')[:2]
        obs = collections.OrderedDict((
            ('gripper_qpos', gripper_qpos),
            ('gripper_qvel', gripper_qvel),
            ('object_pos', object_pos),
            ('target_pos', self.goal_pos),
            ('state_achieved_goal', object_pos.copy()),
            ('state_desired_goal', self.goal_pos.copy()),
        ))
        return obs

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(action, obs) 
        done = False
        return obs, reward, done, {}

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
        return { 'desired_goal': goals, 'state_desired_goal': goals }

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

    def render(self, width=256, height=256, mode='rgb_array'):
        if mode == 'rgb_array':
            if not self.viewer:
                env.initialize_camera(self.camera_init_fn()) 
            img = self.sim.render(width=width, height=height, mode='offscreen')
            return img
        elif mode == 'human':
            super(PusherEnv, self).render(mode=mode) 
        else:
            raise NotImplementedError

# env = PusherEnv()

if __name__ == "__main__":
    env = PusherEnv(reset_gripper=False, reset_object=False)
    for _ in range(100):
        env.reset()
        for _ in range(50):
            env.step(np.random.uniform(-1, 1, size=3))
            img = env.render(mode='human')

            # img = env.render(mode='rgb_array')
            # plt.imshow(img)
            # plt.show()

