import numpy as np
import tensorflow as tf
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

from rlgym.utils.reward_functions.logger import RewardLogger
from rlgym.utils.reward_functions.default_reward import DefaultReward

from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward

from rlgym.utils.gamestates import PlayerData, GameState


if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    agents_per_match = 2
    num_instances = 1
    target_steps = 100_000
    
    print(f"fps={fps}, gamma={gamma})")

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,  # 3v3 to get as many agents going as possible, will make results more noisy
            tick_skip=frame_skip,
            reward_function=CombinedReward(
             (
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=100.0,
                    concede=-100.0,
                    shot=5.0,
                    save=30.0,
                    demo=10.0
                ),
             ),
            (0.2, 1.0, 1.0)),
            spawn_opponents=True,
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()  # Discrete > Continuous don't @ me
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances)            # Start 1 instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    #env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    model = PPO.load(
    "testing/exit_save.zip",
    env,
    device="cuda",
    )
    

    print("Beginning Test:")

    model.learn(1_000_000, progress_bar=True)

    print("Test complete, printing results:")




        
        