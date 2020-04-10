import gym
import numpy as np

from agents.dqn.dqn_agent import Agent as DQNAgent
import envs.manipulator
from collections import deque


def dqn_train(agent, env, n_episodes=1000, max_t=1000, eps_start=0.3, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset(return_all=True)
        score = 0
        for t in range(max_t):
            action = agent.act(np.array(state), eps)
            next_state, reward, done, _ = env.step(action, return_all=True)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 50:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            state = env.reset(return_all=True)
            for t in range(max_t):
                print(state[:4])
                action = agent.act(np.array(state), 0)
                next_state, reward, done, _ = env.step(action, return_all=True)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            break
    return scores


def train(parameters):
    num_episodes = int(parameters['episodes'])
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']
    env_name = "Manipulator-v1"
    situation = {
        'manipulator_angles': [0, 0, 0],
        'grabbed': False,
        'block_pos': 6,
        'task': 'grab'
    }
    env = gym.make(env_name, situation=situation)
    agent = DQNAgent(env.num_of_joints+1+3, env.action_space.n, seed=0)
    dqn_train(agent, env)
    # agent.train(num_episodes, True)
    # policy = q_to_policy(agent.q)
    # agent.environment.build_policy_to_goal(policy=policy)


def q_to_policy(q, offset=0):
    optimal_policy = {}
    for state in q:
        optimal_policy[state] = np.argmax(q[state]) + offset
    return optimal_policy


def main():
    parameters = {'episodes': 100, 'gamma': 0.99, 'alpha': 0.4, 'epsilon': 0.2}
    train(parameters)


if __name__ == '__main__':
    main()
