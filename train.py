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
from multiprocessing import Pool
from copy import deepcopy


def train(parameters):
    print(parameters['bench'], ' starting...')
    num_episodes = int(parameters['episodes'])
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']
    env_name = "BlocksWorld-v1"
    if os.path.exists(parameters['bench']):
        with open(parameters['bench'], 'r') as read:
            map_dict = json.load(read)
    else:
        raise FileNotFoundError
    env = gym.make(env_name, map_dict=map_dict)
    agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
    average_eps_reward, all_rewards = agent.train(num_episodes, False)
    policy = q_to_policy(agent.q)
    if parameters['plot']:
        env.render(policy=policy)

    if parameters['verbose'] or parameters['movement']:
        print(parameters['bench'], ' done!')
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


def train_one_file(path):
    parameters = {'episodes': 1000, 'gamma': 0.99, 'alpha': 0.6, 'epsilon': 0.2,
                  'verbose': True, 'plot': False, 'movement': False, 'bench': path}
    print('---Start---')
    start = time.time()
    average_reward = train(parameters)
    end = time.time()
    print('\nAverage reward: {}', average_reward)
    print('Time (', parameters['episodes'], 'episodes ):', end - start)
    print('---End---')


def train_multiple_files(paths):
    parameters = {'episodes': 1000, 'gamma': 0.99, 'alpha': 0.6, 'epsilon': 0.2,
                  'verbose': True, 'plot': False, 'movement': False, 'bench': ''}
    params_arr = []
    for path in paths:
        curr_params = deepcopy(parameters)
        curr_params['bench'] = path
        params_arr.append(curr_params)
    pool = Pool(processes=len(paths))
    pool.map(train, params_arr)
    pool.close()
    pool.join()


def main():
    path_to_file = 'parsing_jsons/parsed/partial_0/parsed_tasks_0.json'
    files = [f'parsing_jsons/parsed/partial_0/parsed_tasks_{i}.json' for i in range(6)]
    train_multiple_files(files)


if __name__ == '__main__':
    main()
