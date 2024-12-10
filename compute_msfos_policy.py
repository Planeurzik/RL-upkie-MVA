#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria

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

FORCE = 9
CURRENTFORCE = 0
upkie.envs.register()
def parse_command_line_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default="",
        type=str,
        help="name of the new policy to train",
    )
    parser.add_argument(
        "--nb-envs",
        default=1,
        type=int,
        help="number of parallel simulation processes to run",
    )
    parser.add_argument(
        "--show",
        default=False,
        action="store_true",
        help="show simulator during trajectory rollouts",
    )
    return parser.parse_args()



def parse_command_line_arguments() -> argparse.Namespace:
    """!
    Parse command line arguments.

    @returns Command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy",
        nargs="?",
        help="path to the policy parameters file",
    )
    parser.add_argument(
        "--training",
        default=False,
        action="store_true",
        help="add noise and actuation lag, as in training",
    )
    return parser.parse_args()


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

    forces=  np.array(forces)
    successes = np.array(successes)
    sorted_indices = np.argsort(forces)
    forces = forces[sorted_indices]
    successes = successes[sorted_indices]
    plt.title("MSFOS simulation for linear policy")
    plt.plot(forces,successes, label = "proportion of successes for each force applied")
    plt.xlabel("sagital force applied (N)")
    plt.ylabel("proportion of success")
    plt.legend()
    plt.show()


def run_policy(env: gym.Wrapper, policy) -> None:
    """!
    Run the policy on a given environment.

    @param env Upkie environment, wrapped by the agent.
    @param policy MLP policy to follow.
    """
    global FORCE, CURRENTFORCE
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

    bullet_action = {
            "external_forces": {
                "torso": {
                    "force": np.array([FORCE,0,0]),
                    "local": False,
                }
            }
        }
    count = 0

    observations = []
    forces = []
    gain = np.array ([20.0 , 1.0 , 1.0 , 0.1,0.])
    while True:
        action, _ = policy.predict(observation, deterministic=True)
        #action = np.clip(gain.dot(observation[-1].reshape((-1,))),-1,1)
        tip_position, tip_velocity = get_tip_state(observation[-1])
        env.unwrapped.log("action", action)
        env.unwrapped.log("observation", observation[-1])
        env.unwrapped.log("reward", reward)
        env.unwrapped.log("tip_position", tip_position)
        env.unwrapped.log("tip_velocity", tip_velocity)
        if count>300 and count < 500:
            print("applying force !!!")
            env.bullet_extra(bullet_action)
            CURRENTFORCE = FORCE
        else:
            CURRENTFORCE = 0
            env.bullet_extra(bullet_no_action)
        forces.append(CURRENTFORCE)
        observations.append(observation[0][0:2])
        observation, reward, terminated, truncated, info = env.step(action)
        print(info['spine_observation']['sim']['base']['position'])

        count+=1
        if terminated or truncated or count > 2000:
            break
            count = 0
            observation, info = env.reset()
    time = np.arange(len(observations))*1/200
    observations = np.array(observations)
    forces=  np.array(forces)
    plt.title("MSFOS simulation for base ppo policy")
    plt.plot(time,observations[:,0], label = "angle (rad)")
    plt.plot(time,observations[:,1], label = "position (m)")
    plt.plot(time,forces/10, label="force ( 1 unit = 10 N )  applied")
    plt.legend()
    plt.show()


def main(policy_path: str, training: bool) -> None:
    """!
    Load environment and policy, and run the latter on the former.

    @param policy_path Path to policy parameters.
    @param training If True, add training noise and domain randomization.
    """
    env_settings = EnvSettings()

    init_state = None
    if training:
        training_settings = TrainingSettings()
        init_state = RobotState(
            randomization=RobotStateRandomization(
                **training_settings.init_rand
            ),
        )
    with gym.make(
        env_settings.env_id,
        frequency=env_settings.agent_frequency,
        init_state=init_state,
        max_ground_velocity=env_settings.max_ground_velocity,
        regulate_frequency=True,
        spine_config=env_settings.spine_config,
    ) as velocity_env:
        env = make_ppo_balancer_env(
            velocity_env,
            env_settings,
            training=training,
        )

         
        ppo_settings = PPOSettings()
        policy = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={
                "net_arch": {
                    "pi": ppo_settings.net_arch_pi,
                    "vf": ppo_settings.net_arch_vf,
                },
            },
            verbose=0,
        )
        policy.set_parameters(policy_path)
        policy.save(f"./final_converted.zip")
        #run_policy(env, policy)
        compute_msfos(env, policy)


if __name__ == "__main__":
    if on_raspi():
        configure_agent_process()

    agent_dir = os.path.abspath(os.path.dirname(__file__))
    args = parse_command_line_arguments()

    # Policy parameters
    policy_path = args.policy
    if policy_path is None:
        policy_path = f"{agent_dir}/policy/params.zip"
    if policy_path.endswith(".zip"):
        policy_path = policy_path[:-4]
    logging.info("Loading policy from %s.zip", policy_path)

    # Configuration
    config_path = f"{os.path.dirname(policy_path)}/operative_config.gin"
    logging.info("Loading policy configuration from %s", config_path)
    gin.parse_config_file(config_path)

    try:
        main(policy_path, args.training)
    except KeyboardInterrupt:
        logging.info("Caught a keyboard interrupt")
