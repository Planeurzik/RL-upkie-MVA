import argparse
import logging
import os
from typing import Tuple

import gin
import gymnasium as gym
import numpy as np
import upkie.envs
from envs import make_ppo_balancer_env
from settings import EnvSettings, PPOSettings, TrainingSettings
from stable_baselines3 import PPO
from upkie.utils.raspi import configure_agent_process, on_raspi
from upkie.utils.robot_state import RobotState
from upkie.utils.robot_state_randomization import RobotStateRandomization
import matplotlib.pyplot as plt

def get_tip_state(
    observation, tip_height: float = 0.58
) -> Tuple[float, float]:
    """!
    Compute the state of the virtual tip used in the agent's reward.

    This extra info is for logging only.

    @param observation Observation vector.
    @param tip_height Height of the virtual tip.
    @returns Pair of tip (position, velocity) in the sagittal planeg
    """
    pitch = observation[0]
    ground_position = observation[1]
    angular_velocity = observation[2]
    ground_velocity = observation[3]
    tip_position = ground_position + tip_height * np.sin(pitch)
    tip_velocity = ground_velocity + tip_height * angular_velocity * np.cos(
        pitch
    )
    return tip_position, tip_velocity

def compute_msfos(env: gym.Wrapper, policy) -> None:
    """!
    Run the policy on a given environment.

    @param env Upkie environment, wrapped by the agent.
    @param policy MLP policy to follow.
    """
    action = np.zeros(env.action_space.shape)
    observation, info = env.reset()
    reward = 0.0
    bullet_no_action = {
            "external_forces": {
                "torso": {
                    "force": np.array([0.,0.,0.0]),
                    "local": False,
                }
            }
        }


    observations = []
    forces = []
    forcemax = 21
    forcemin = 0
    n_tries = 3
    successes = []
    for i in range(5):
        nbsuccess = 0
        force = forcemax+forcemin
        force /= 2
        bullet_action = {
            "external_forces": {
                "torso": {
                    "force": np.array([force,0.0,0]),
                    "local": False,
                }
            }
        }

        for n in range(n_tries):
            count = 0
            observation, info = env.reset()
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                tip_position, tip_velocity = get_tip_state(observation[-1])
                env.unwrapped.log("action", action)
                env.unwrapped.log("observation", observation[-1])
                env.unwrapped.log("reward", reward)
                env.unwrapped.log("tip_position", tip_position)
                env.unwrapped.log("tip_velocity", tip_velocity)
                if count>300 and count < 500:
                    print("applying force !!!")
                    env.bullet_extra(bullet_action)
                else:
                    env.bullet_extra(bullet_no_action)
                observation, reward, terminated, truncated, info = env.step(action)

                count+=1
                if count > 2000:
                    nbsuccess +=1
                    break

                if terminated or truncated:
                    break

        if nbsuccess< n_tries:
            forcemax = force
        else:
            forcemin=  force
        print(forcemin,forcemax, nbsuccess)
            
        successes.append(nbsuccess/n_tries)
        forces.append(force)