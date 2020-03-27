import pickle
import os.path as osp
import os
import glob
import pickle
import gzip
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv

from serializable import Serializable

from softlearning.misc.utils import PROJECT_PATH
from softlearning.environments.helpers import random_point_in_circle

class PusherEnv(MujocoEnv, Serializable):

    MODEL_PATH = os.path.join(
        os.path.dirname(multiworld.envs.__file__),
        'assets/pusher2d/pusher2d_simple.xml')

    def __init__(self, do_reset=True, multi_reset=False, reset_block=True, reset_gripper=True, multi_reset_block=False):
        self.do_reset = do_reset
        self.multi_reset = multi_reset
        self.multi_reset_block = multi_reset_block
        self.reset_block = reset_block
        self.reset_gripper = reset_gripper
        self._Serializable__initialize(locals())
        self.reset_already = False
        self.goal_pos = np.array([2.0, 0.])
        MujocoEnv.__init__(self, model_path=self.MODEL_PATH, frame_skip=5)
        self.model.stat.extent = 10

    def reset(self, reset_args=None):
        goal_positions = np.array([[2.0, 2.0],
                                   # [2.0, -2.0],
                                   # [-2.0, 2.0],
                                   # [-2.0, -2.0]
                                   ])
        if self.reset_already is False or self.do_reset:
            qpos = np.zeros((5,))
            qvel = np.zeros((5,))
            #Block resetting
            if self.reset_block or self.reset_already is False:
                qpos[3] = 1.0
                qpos[4] = 0.0
            elif self.multi_reset_block:
                qpos[3] = np.random.uniform(-1, 1)
                qpos[4] = np.random.uniform(-1, 1)
            else:
                qpos[3] = self._get_obs()[3]
                qpos[4] = self._get_obs()[4]

            #Gripper resetting
            if self.multi_reset:
                xp = np.random.uniform(-1.5, 1.5, size=(2, ))
                qpos[:2] = xp
            elif self.reset_gripper:
                qpos[0] = 0
                qpos[1] = 0
                qpos[2] = 0
            elif self.reset_gripper is False:
                qpos[0] = self._get_obs()[0]
                qpos[1] = self._get_obs()[1]
                qpos[2] = self._get_obs()[2]
            self.set_state(np.array(qpos), np.array(qvel))
            self.reset_already = True
        goal_idx = np.random.randint(len(goal_positions))
        self.goal_pos = goal_positions[goal_idx]
        target_id = self.model.body_name2id('target')
        bp = self.model.body_pos.copy()
        bp[target_id, :2] = self.goal_pos
        self.model.body_pos[:] = bp
        self.sim.forward()
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:3],
            self.sim.data.geom_xpos[-2:-1, :2].flat,
            self.sim.data.qvel.flat,
            self.goal_pos
        ]).reshape(-1)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs()

        curr_block_xidx = 3
        curr_block_yidx = 4
        curr_gripper_pos = self.sim.data.site_xpos[0, :2]
        curr_block_pos = np.array([next_obs[curr_block_xidx], next_obs[curr_block_yidx]])


        dist_to_block = np.linalg.norm(curr_gripper_pos - curr_block_pos)
        block_dist = np.linalg.norm(self.goal_pos - curr_block_pos)

        if dist_to_block > 0.3:
            reward = -block_dist
        else:
            reward = -dist_to_block + 10
        # reward = - dist_to_block - 10*block_dist

        done = False
        self.render()
        return next_obs, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[:3] = [0, 0, 0]
        self.viewer.cam.distance = 5.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0
        self.viewer.cam.trackbodyid = -1
    #
    # def render(self, width=32, height=32, camera_id=-1, mode='rgb_array'):
    #     img = self.sim.render(width=width, height=height, mode=mode)
    #     return img

if __name__ == "__main__":
    env = PusherEnv()
    import IPython
    IPython.embed()
