import time
import gym
import numpy as np
import envs.gridworld
import matplotlib.pyplot as plt
from utils.option_methods import load_option
from agents.qlearning.qlearning_agent import QLearningAgent, QLearningWithOptionsAgent


def train(parameters, with_options=False, intra_options=False):
    num_episodes = int(parameters['episodes'])
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']
    env_name = "GridWorld-v1"
    env = gym.make(env_name)

    if with_options:
        options = [load_option('GridWorldOption1')]
        agent = QLearningWithOptionsAgent(env, options, gamma=gamma, alpha=alpha, epsilon=epsilon,
                                          intra_options=intra_options)
    else:
        agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    average_eps_reward, all_rewards = agent.train(num_episodes)
    env.render(draw_arrows=True, policy=q_to_policy(agent.q),
               name_prefix=env_name)
    # plt.xlabel('iteration')
    # plt.ylabel('reward')
    # plt.plot(all_rewards[-100:])
    # plt.show()

    env.close()
    return average_eps_reward


def q_to_policy(q, offset=0):
    optimalPolicy = {}
    for state in q:
        optimalPolicy[state] = np.argmax(q[state]) + offset
    return optimalPolicy


def main():
    parameters = {'episodes': 1000, 'gamma': 0.9, 'alpha': 0.1, 'epsilon': 0.1}
    print('---Start---')
    start = time.time()
    average_reward = train(parameters, with_options=False, intra_options=False)
    end = time.time()
    print('\nAverage reward: {}', average_reward)
    print('Time (', parameters['episodes'], 'episodes ):', end - start)
    print('---End---')


if __name__ == '__main__':
    main()
