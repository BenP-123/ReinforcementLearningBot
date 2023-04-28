import numpy as np
import tensorflow as tf
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan
from stable_baselines3.ppo.policies import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward

from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure


if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    agents_per_match = 2
    num_instances = 1   #try more later?
    target_steps = 100_000
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps
    


    print(f"fps={fps}, gamma={gamma})")


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,  # 3v3 to get as many agents going as possible, will make results more noisy
            tick_skip=frame_skip,
            game_speed=1,
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
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()], 
            obs_builder=AdvancedObs(),  
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances)            # Start 1 instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    try:
        model = PPO.load(
        "models/exit_save.zip",
        env,
        device="auto",  # Need to set device again (if using a specific one)
        # force_reset=True  # Make SB3 reset the env so it doesn't think we're continuing from last state   dont think this is needed
        )
    except:
        print("Starting new model:")
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
        )

                        #hyperps
        model = PPO(
            MlpPolicy,
            env,
            n_epochs=1,                 # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=2,                   # Print out all the info as we're going
            batch_size=batch_size,             # Batch size as high as possible within reason
            n_steps=steps,                # Number of steps to perform before optimizing network
            tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="cuda"                # Uses GPU if available
        )

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(10_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")

    tmp_path = "A:/Capstone/PyBot/logs/FancyLogs"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    print("Entering While loop:")

    while True:
        model.set_logger(new_logger)
        model.learn(20_000_000, callback=callback)
        model.save("models/exit_save")
        model.save(f"mmr_models/{model.num_timesteps}")