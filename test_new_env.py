import gym
import numpy as np

from agents.qlearning.qlearning_agent import QLearningAgent
import envs.manipulator


def train(parameters):
    num_episodes = int(parameters['episodes'])
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']
    env_name = "Manipulator-v1"
    situation = {
        'manipulator_angles': [270, 0, 0],
        'grabbed': False,
        'block_pos': 3,
        'task': 'grab'
    }
    env = gym.make(env_name, situation=situation)
    agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
    average_eps_reward, all_rewards = agent.train(num_episodes, True)
    policy = q_to_policy(agent.q)
    agent.environment.build_policy_to_goal(policy=policy)
    return average_eps_reward


def q_to_policy(q, offset=0):
    optimal_policy = {}
    for state in q:
        optimal_policy[state] = np.argmax(q[state]) + offset
    return optimal_policy


def main():
    parameters = {'episodes': 100, 'gamma': 0.9, 'alpha': 0.4, 'epsilon': 0.2}
    train(parameters)


if __name__ == '__main__':
    main()
