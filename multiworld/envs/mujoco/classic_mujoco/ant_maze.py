import numpy as np

from multiworld.envs.mujoco.classic_mujoco.ant import AntEnv


class AntMazeEnv(AntEnv):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.quick_init(locals())
        AntEnv.__init__(self, *args, **kwargs)

        if self.model_path == 'classic_mujoco/ant_maze2_gear30_small_dt3.xml':
            self.maze_type = 'u-small'
        elif self.model_path == 'classic_mujoco/ant_maze2_gear30_big_dt3.xml':
            self.maze_type = 'u-big'
        elif self.model_path == 'classic_mujoco/ant_fb_gear30_small_dt3.xml':
            self.maze_type = 'fb-small'
        elif self.model_path == 'classic_mujoco/ant_fb_gear30_med_dt3.xml':
            self.maze_type = 'fb-med'
        elif self.model_path == 'classic_mujoco/ant_fb_gear30_big_dt3.xml':
            self.maze_type = 'fb-big'
        elif self.model_path == 'classic_mujoco/ant_fork_gear30_med_dt3.xml':
            self.maze_type = 'fork-med'
        elif self.model_path == 'classic_mujoco/ant_fork_gear30_big_dt3.xml':
            self.maze_type = 'fork-big'
        else:
            raise NotImplementedError

        if self.maze_type == 'u-small':
            self.walls = [
                Wall(0, 1.125, 1.25, 2.375, self.ant_radius),

                Wall(0, 4.5, 3.5, 1, self.ant_radius),
                Wall(0, -4.5, 3.5, 1, self.ant_radius),
                Wall(4.5, 0, 1, 5.5, self.ant_radius),
                Wall(-4.5, 0, 1, 5.5, self.ant_radius),
            ]
        elif self.maze_type == 'fb-small':
            self.walls = [
                Wall(-2.0, 1.25, 0.75, 4.0, self.ant_radius),
                Wall(2.0, -1.25, 0.75, 4.0, self.ant_radius),

                Wall(0, 6.25, 5.25, 1, self.ant_radius),
                Wall(0, -6.25, 5.25, 1, self.ant_radius),
                Wall(6.25, 0, 1, 7.25, self.ant_radius),
                Wall(-6.25, 0, 1, 7.25, self.ant_radius),
            ]
        elif self.maze_type == 'fb-med':
            self.walls = [
                Wall(-2.25, 1.5, 0.75, 4.5, self.ant_radius),
                Wall(2.25, -1.5, 0.75, 4.5, self.ant_radius),

                Wall(0, 7.0, 6.0, 1, self.ant_radius),
                Wall(0, -7.0, 6.0, 1, self.ant_radius),
                Wall(7.0, 0, 1, 8.0, self.ant_radius),
                Wall(-7.0, 0, 1, 8.0, self.ant_radius),
            ]
        elif self.maze_type == 'fb-big':
            self.walls = [
                Wall(-2.75, 2.0, 0.75, 5.5, self.ant_radius),
                Wall(2.75, -2.0, 0.75, 5.5, self.ant_radius),

                Wall(0, 8.5, 7.5, 1, self.ant_radius),
                Wall(0, -8.5, 7.5, 1, self.ant_radius),
                Wall(8.5, 0, 1, 9.5, self.ant_radius),
                Wall(-8.5, 0, 1, 9.5, self.ant_radius),
            ]
        elif self.maze_type == 'fork-med':
            self.walls = [
                Wall(-1.75, -1.5, 0.25, 3.5, self.ant_radius),
                Wall(0, 1.75, 2.0, 0.25, self.ant_radius),
                Wall(0, -1.75, 2.0, 0.25, self.ant_radius),

                Wall(0, 6.0, 5.0, 1, self.ant_radius),
                Wall(0, -6.0, 5.0, 1, self.ant_radius),
                Wall(6.0, 0, 1, 7.0, self.ant_radius),
                Wall(-6.0, 0, 1, 7.0, self.ant_radius),
            ]
        elif self.maze_type == 'fork-big':
            self.walls = [
                Wall(-3.5, -1.5, 0.25, 5.25, self.ant_radius),
                Wall(0, -3.5, 3.75, 0.25, self.ant_radius),
                Wall(0, 0.0, 3.75, 0.25, self.ant_radius),
                Wall(0, 3.5, 3.75, 0.25, self.ant_radius),

                Wall(0, 7.75, 6.75, 1, self.ant_radius),
                Wall(0, -7.75, 6.75, 1, self.ant_radius),
                Wall(7.75, 0, 1, 8.75, self.ant_radius),
                Wall(-7.75, 0, 1, 8.75, self.ant_radius),
            ]
        else:
            raise NotImplementedError


    def _collision_idx(self, pos):
        bad_pos_idx = []
        for i in range(len(pos)):
            for wall in self.walls:
                if wall.contains_point(pos[i]):
                    bad_pos_idx.append(i)
                    break

            # if 'small' in self.model_path:
            #     if 'maze2' in self.model_path:
            #         if (-2.00 <= pos[i][0] <= 2.00) and (-2.00 <= pos[i][1]):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][0]) or (pos[i][0] <= -2.75):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][1]) or (pos[i][1] <= -2.75):
            #             bad_pos_idx.append(i)
            #     else:
            #         if (-2.00 <= pos[i][0] <= 2.00) and (-2.00 <= pos[i][1] <= 2.00):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][0]) or (pos[i][0] <= -2.75):
            #             bad_pos_idx.append(i)
            #         elif (2.75 <= pos[i][1]) or (pos[i][1] <= -2.75):
            #             bad_pos_idx.append(i)
            # elif 'big' in self.model_path:
            #     if 'maze2' in self.model_path:
            #         if (-2.25 <= pos[i][0] <= 2.25) and (-2.75 <= pos[i][1]):
            #             bad_pos_idx.append(i)
            #         elif (4.75 <= pos[i][0]) or (pos[i][0] <= -4.75):
            #             bad_pos_idx.append(i)
            #         elif (4.75 <= pos[i][1]) or (pos[i][1] <= -4.75):
            #             bad_pos_idx.append(i)
            #     else:
            #         raise NotImplementedError
            # else:
            #     raise NotImplementedError

        return bad_pos_idx

    def _sample_uniform_xy(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low[:2],
            self.goal_space.high[:2],
            size=(batch_size, 2),
        )

        bad_goals_idx = self._collision_idx(goals)
        goals = np.delete(goals, bad_goals_idx, axis=0)
        while len(bad_goals_idx) > 0:
            new_goals = np.random.uniform(
                self.goal_space.low[:2],
                self.goal_space.high[:2],
                size=(len(bad_goals_idx), 2),
            )

            bad_goals_idx = self._collision_idx(new_goals)
            new_goals = np.delete(new_goals, bad_goals_idx, axis=0)
            goals = np.concatenate((goals, new_goals))

        # if 'small' in self.model_path:
        #     goals[(0 <= goals) * (goals < 0.5)] += 1
        #     goals[(0 <= goals) * (goals < 1.25)] += 1
        #     goals[(0 >= goals) * (goals > -0.5)] -= 1
        #     goals[(0 >= goals) * (goals > -1.25)] -= 1
        # else:
        #     goals[(0 <= goals) * (goals < 0.5)] += 2
        #     goals[(0 <= goals) * (goals < 1.5)] += 1.5
        #     goals[(0 >= goals) * (goals > -0.5)] -= 2
        #     goals[(0 >= goals) * (goals > -1.5)] -= 1.5
        return goals

class Wall:
    def __init__(self, x_center, y_center, x_thickness, y_thickness, min_dist):
        self.min_x = x_center - x_thickness - min_dist
        self.max_x = x_center + x_thickness + min_dist
        self.min_y = y_center - y_thickness - min_dist
        self.max_y = y_center + y_thickness + min_dist

    def contains_point(self, point):
        return (self.min_x < point[0] < self.max_x) and (self.min_y < point[1] < self.max_y)

    def contains_points(self, points):
        return (self.min_x < points[:,0]) * (points[:,0] < self.max_x) \
               * (self.min_y < points[:,1]) * (points[:,1] < self.max_y)


if __name__ == '__main__':
    env = AntMazeEnv(
        goal_low=[-4, -4],
        goal_high=[4, 4],
        goal_is_xy=True,
        reward_type='xy_dense',
    )
    import gym
    from multiworld.envs.mujoco import register_custom_envs
    register_custom_envs()
    env = gym.make('AntMaze150RandomInitEnv-v0')
    # env = gym.make('AntCrossMaze150Env-v0')
    # env = gym.make('DebugAntMaze30BottomLeftRandomInitGoalsPreset1Env-v0')
    env = gym.make(
        # 'AntMaze30RandomInitFS20Env-v0',
        # 'AntMaze30RandomInitEnv-v0',
        # 'AntMazeSmall30RandomInitFS10Env-v0',
        # 'AntMazeSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMaze30RandomInitNoVelEnv-v0',
        # 'AntMaze30StateEnv-v0',
        # 'AntMaze30QposRandomInitFS20Env-v0',
        # 'AntMazeSmall30RandomInitFs10Dt3Env-v0',
        # 'AntMazeQposRewSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMazeXyRewSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMazeQposRewSmall30Fs5Dt3Env-v0',
        # 'AntMazeQposRewSmall30Fs5Dt3NoTermEnv-v0',
        # 'AntMazeXyRewSmall30RandomInitFs5Dt3NoTermEnv-v0',
        # 'AntMazeXyRewSmall30Fs5Dt3NoTermEnv-v0',
        'AntMazeQposRewSmall30Fs5Dt3NoTermEnv-v0',
    )
    env.reset()
    i = 0
    while True:
        i += 1
        env.render()
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        obs, reward, done, info = env.step(action)
        # print(reward, np.linalg.norm(env.sim.data.get_body_xpos('torso')[:2]
        #                              - env._xy_goal) )
        # print(env.sim.data.qpos)
        print(info)
        if i % 5 == 0:
            env.reset()
