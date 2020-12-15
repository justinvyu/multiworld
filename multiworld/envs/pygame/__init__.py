from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)
REGISTERED = False


def register_pygame_envs():
    global REGISTERED
    if REGISTERED:
        return
    REGISTERED = True
    LOGGER.info("Registering multiworld pygame gym environments")

    # === 2D Point Mass (No Walls) ===
    register(
        id='Point2DFixed-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
    )

    # === 2D Point Mass (With Walls) ===
    register(
        id='Point2DSingleWall-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'wall_shape': '--',
            'wall_thickness': 2.0,
            'render_onscreen': True,
            'images_are_rgb': True,
            'render_target': True,
            'inner_wall_max_dist': 2,
        },
    )
    register(
        id='Point2DBoxWall-v1',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'wall_shape': 'box',
            'wall_thickness': 2.0,
            # 'render_size': 84,
            'render_onscreen': True,
            'images_are_rgb': True,
            'render_target': True,
        },
    )
    register(
        id='Point2DMazeEvalHard-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'wall_shape': 'hard-maze',
            'reward_type': 'sparse',
            'init_pos_range': ([-3, -3], [-3, -3]),
            'render_onscreen': False,
            'boundary_dist': 4,
            'inner_wall_max_dist': 2,
            'images_are_rgb': True,
            'render_target': True,
            'fix_goal_position': True,
            'goal_position': [-1, 1],
            'action_scale': 0.5,
        },
    )
    register(
        id='Point2DMazeEvalMedium-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'wall_shape': 'medium-maze',
            'reward_type': 'sparse',
            'init_pos_range': ([-3, -3], [-3, -3]),
            'render_onscreen': False,
            'boundary_dist': 4,
            'inner_wall_max_dist': 2,
            'images_are_rgb': True,
            'render_target': True,
            'fix_goal_position': True,
            'goal_position': [3, 3],
            'action_scale': 0.5,
        },
    )
    register(
        id='Point2DMazeEvalEasy-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'wall_shape': 'easy-maze',
            'reward_type': 'sparse',
            'init_pos_range': ([-3, -3], [-3, -3]),
            'render_onscreen': False,
            'boundary_dist': 4,
            'inner_wall_max_dist': 2,
            'images_are_rgb': True,
            'render_target': True,
            'fix_goal_position': True,
            'goal_position': [3, -3],
            'action_scale': 0.5,
        },
    )
    register(
        id='Point2DRooms-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'wall_shape': 'rooms',
            'reward_type': 'sparse',
            'init_pos_range': ([-3, -3], [-3, -3]),
            'render_onscreen': False,
            'boundary_dist': 4,
            'inner_wall_max_dist': 2,
            'images_are_rgb': True,
            'render_target': True,
            'fix_goal_position': True,
            'goal_position': [3, 3],
            'action_scale': 0.5,
        },
    )
    register(
        id='Point2DMazeExplore-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'wall_shape': 'hard-maze',
            'reward_type': 'sparse',
            'init_pos_range': ([-3, -3], [-3, -3]),
            'render_onscreen': False,
            'boundary_dist': 4,
            'inner_wall_max_dist': 2,
            'images_are_rgb': True,
            'render_target': True
        }
    )
    register(
        id='Point2DDoubleMazeExplore-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'wall_shape': 'double-maze',
            'reward_type': 'sparse',
            'target_radius': 1,
            'ball_radius': 0.3,
            'init_pos_range': ([0, 0], [0, 0]),
            'render_onscreen': False,
            'boundary_dist': 16,
            'inner_wall_max_dist': 2,
            'images_are_rgb': True,
            'render_target': True,
        }
    )
    register(
        id='Point2DDoubleMazeEval-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'wall_shape': 'double-maze',
            'reward_type': 'sparse',
            'target_radius': 1,
            'ball_radius': 0.3,
            'init_pos_range': ([0, 0], [0, 0]),
            'render_onscreen': False,
            'boundary_dist': 16,
            'inner_wall_max_dist': 2,
            'images_are_rgb': True,
            'render_target': True,
            'fix_goal_position': True,
            'multiple_goals': True,
            'goal_position': [[-9, -3], [9, 3]],
        }
    )
    register(
        id='Point2DDoubleMazeSingleGoalEval-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        kwargs={
            'wall_shape': 'double-maze',
            'reward_type': 'sparse',
            'target_radius': 1,
            'ball_radius': 0.3,
            'init_pos_range': ([0, 0], [0, 0]),
            'render_onscreen': False,
            'boundary_dist': 16,
            'inner_wall_max_dist': 2,
            'images_are_rgb': True,
            'render_target': True,
            'fix_goal_position': True,
            'multiple_goals': False,
            'goal_position': [-9, -3],
        }
    )
    register(
        id='Point2D-Box-Wall-v1',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        tags={
            'git-commit-hash': '73c8823',
            'author': 'vitchyr'
        },
        kwargs={
            'action_scale': 1.,
            'wall_shape': 'box',
            'wall_thickness': 2.0,
            # 'render_size': 84,
            'render_onscreen': True,
            'images_are_rgb': True,
            'render_target': True,
        },
    )
    register(
        id='Point2D-Big-UWall-v1',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        tags={
            'git-commit-hash': '73c8823',
            'author': 'vitchyr'
        },
        kwargs={
            'action_scale': 0.25,
            'wall_shape': 'big-u',
            'wall_thickness': 0.50,
            'render_size': 84,
            'images_are_rgb': True,
            'render_onscreen': True,
            'render_target': True,
        },
    )
    register(
        id='Point2D-Easy-UWall-v1',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        tags={
            'git-commit-hash': '73c8823',
            'author': 'vitchyr'
        },
        kwargs={
            'action_scale': 0.25,
            'wall_shape': 'easy-u',
            'wall_thickness': 0.50,
            'render_size': 84,
            'images_are_rgb': True,
            'render_onscreen': True,
            'render_target': True,
        },
    )

    # === Point Mass Image Environments ===
    register(
        id='Point2DEnv-ImageFixedGoal-v0',
        entry_point=point2d_image_fixed_goal_v0,
        tags={
            'git-commit-hash': '2e92a51',
            'author': 'vitchyr'
        },
    )
    register(
        id='Point2DEnv-Image-v0',
        entry_point=point2d_image_v0,
        tags={
            'git-commit-hash': '78c9f9e',
            'author': 'vitchyr'
        },
    )


def point2d_image_fixed_goal_v0(**kwargs):
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DEnv
    from multiworld.core.flat_goal_env import FlatGoalEnv
    env = Point2DEnv(
        fixed_goal=(0, 0),
        images_are_rgb=True,
        render_onscreen=False,
        show_goal=True,
        ball_radius=2,
        render_size=8,
    )
    env = ImageEnv(env, imsize=env.render_size, transpose=True)
    env = FlatGoalEnv(env)
    return env


def point2d_image_v0(**kwargs):
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DEnv
    env = Point2DEnv(
        images_are_rgb=True,
        render_onscreen=False,
        show_goal=True,
        ball_radius=1,
        render_size=100,
    )
    env = ImageEnv(env, imsize=env.render_size, transpose=True)
    return env
