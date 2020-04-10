import json
import os
import sys
import time
from copy import deepcopy
from multiprocessing import Pool

import gym
import numpy as np
import torch

from agents.qlearning.qlearning_agent import QLearningAgent
from agents.dqn.dqn_agent import Agent as DQNAgent
from map_spatial_wrapper.test2 import main as planner_main
import envs.blocks
import envs.manipulator
from planner_parser import parse


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

    if parameters['verbose'] or parameters['movement'] or parameters['save_path'] is not None:
        agent.environment.build_policy_to_goal(policy=policy,
                                               movement=parameters['movement'],
                                               verbose=parameters['verbose'],
                                               save_path=parameters['save_path'])
    env.close()
    print(parameters['bench'], ' done!')
    return average_eps_reward


def q_to_policy(q, offset=0):
    optimal_policy = {}
    for state in q:
        optimal_policy[state] = np.argmax(q[state]) + offset
    return optimal_policy


def train_one_file(parameters):
    print('---Start---')
    start = time.time()
    average_reward = train(parameters)
    end = time.time()
    print('\nAverage reward: {}', average_reward)
    print('Time (', parameters['episodes'], 'episodes ):', end - start)
    print('---End---')


def train_rl_multiple_files(paths, parameters):
    params_arr = []
    for i, path in enumerate(paths):
        curr_params = deepcopy(parameters)
        curr_params['bench'] = path
        curr_params['save_path'] += f'rl_output_{i}.json'
        params_arr.append(curr_params)
    pool = Pool(processes=len(paths))
    pool.map(train, params_arr)
    pool.close()
    pool.join()


def apply_manipulator_model(situations_path, to_path):
    with open(situations_path + 'situations.json', 'r') as read:
        situations = json.load(read)
    env_name = "Manipulator-v1"
    current_situation = situations[0]
    for i in range(len(situations)):
        env = gym.make(env_name, situation=current_situation)
        agent = DQNAgent(env.num_of_joints + 1 + 1, env.action_space.n, seed=0)
        agent.qnetwork_local.load_state_dict(torch.load('agents/dqn/models/model.pth'))
        state = env.reset(return_all=True)
        solution = []
        for t in range(100):
            action = agent.act(np.array(state), 0)
            next_state, reward, done, _ = env.step(action, return_all=True)
            agent.step(state, action, reward, next_state, done)
            curr_state = {
                'manipulator': env.denormalize(state[:env.num_of_joints]),
                'grabbed': bool(state[env.num_of_joints]),
                'action': env.action_to_str(action)
            }
            solution.append(curr_state)
            state = next_state
            if done:
                break
        with open(to_path + f'{current_situation["id"]}.json', 'w+') as write:
            write.write(json.dumps(solution, indent=4))
        new_man_state = env.denormalize(state[:env.num_of_joints])
        if i != len(situations)-1:
            current_situation = situations[i + 1]
            current_situation['manipulator_angles'] = new_man_state


def extract_situations(from_path, to_path):
    situations = []
    situation_template = {
        'manipulator_angles': [0, 0, 0],
        'grabbed': False,
        'block_pos': 0,
        'task': 'grab'
    }
    id = 0
    for name in sorted(os.listdir(from_path), key=lambda file: int(file.split('.')[0].split('_')[-1])):
        filename = from_path + name
        with open(filename, 'r') as read:
            steps = json.load(read)
        for step in steps:
            if 'block_pos' in step:
                grab = step['action'] == 'pick up'
                curr_sit = deepcopy(situation_template)
                curr_sit['grabbed'] = not grab
                curr_sit['task'] = 'grab' if grab else 'release'
                curr_sit['block_pos'] = step['block_pos']
                curr_sit['id'] = id
                situations.append(curr_sit)
                id += 1
    with open(to_path+'situations.json', 'w+') as write:
        write.write(json.dumps(situations, indent=4))


def train_planner(task_num):
    planner_main(sys.argv[1:], str(task_num), f'tasks_jsons/task{task_num}/planner_steps/')


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    task_num = '0'
    path_prefix = f'tasks_jsons/task{task_num}/'
    planner_steps_path = path_prefix + 'planner_steps/'
    planner_steps_parsed_path = path_prefix + 'planner_steps_parsed/'
    rl_agent_steps_path = path_prefix + f'rl_agent_steps/'
    manipulator_situations_path = path_prefix + 'manipulator_sits_raw/'
    manipulator_situations_solved_path = path_prefix + 'manipulator_sits_solved/'
    create_dir(path_prefix)
    create_dir(planner_steps_path)
    create_dir(planner_steps_parsed_path)
    create_dir(rl_agent_steps_path)
    create_dir(manipulator_situations_path)
    create_dir(manipulator_situations_solved_path)
    train_planner(task_num)
    print('PLANNER FINISHED, PARSING TO RL STARTED')
    parse(planner_steps_path, planner_steps_parsed_path, multiple=True, window_size=20)
    print('PARSING TO RL FINISHED, RL LEARNING STARTED')
    parsed_tasks_len = len(os.listdir(planner_steps_parsed_path))
    tasks_files = [planner_steps_parsed_path + f'parsed_tasks_{i}.json'
                   for i in range(parsed_tasks_len)]
    parameters = {'episodes': 1000, 'gamma': 0.99, 'alpha': 0.6, 'epsilon': 0.2,
                  'verbose': False, 'plot': False, 'movement': False, 'bench': '',
                  'save_path': rl_agent_steps_path}
    train_rl_multiple_files(tasks_files, parameters)
    extract_situations(rl_agent_steps_path, manipulator_situations_path)
    apply_manipulator_model(manipulator_situations_path, manipulator_situations_solved_path)


if __name__ == '__main__':
    main()
