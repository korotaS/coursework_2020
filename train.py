import time
import gym
import numpy as np
import envs.gridworld
import envs.blocks
import matplotlib.pyplot as plt
from utils.option_methods import load_option
from agents.qlearning.qlearning_agent import QLearningAgent, QLearningWithOptionsAgent
import json
import os


def train(parameters):
    num_episodes = int(parameters['episodes'])
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']
    env_name = "BlocksWorld-v1"
    if os.path.exists('benchmarks/' + parameters['bench']):
        with open('benchmarks/' + parameters['bench'], 'r') as read:
            map_dict = json.load(read)
    else:
        raise FileNotFoundError
    env = gym.make(env_name, map_dict=map_dict)
    agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
    average_eps_reward, all_rewards = agent.train(num_episodes)
    policy = q_to_policy(agent.q)
    if parameters['plot']:
        env.render(policy=policy)

    if parameters['verbose'] or parameters['movement']:
        print()
        agent.environment.build_policy_to_goal(policy=policy,
                                               movement=parameters['movement'],
                                               verbose=parameters['verbose'])
    env.close()
    return average_eps_reward


def q_to_policy(q, offset=0):
    optimalPolicy = {}
    for state in q:
        optimalPolicy[state] = np.argmax(q[state]) + offset
    return optimalPolicy


def main():
    parameters = {'episodes': 1000, 'gamma': 0.95, 'alpha': 0.5, 'epsilon': 0.2,
                  'verbose': True, 'plot': False, 'movement': False, 'bench': 'blocks/task3.json'}
    print('---Start---')
    start = time.time()
    average_reward = train(parameters)
    end = time.time()
    print('\nAverage reward: {}', average_reward)
    print('Time (', parameters['episodes'], 'episodes ):', end - start)
    print('---End---')


if __name__ == '__main__':
    main()
