import itertools
import sys
from collections import defaultdict
from ast import literal_eval

import numpy as np

from agents.agent import Agent


class QLearningAgent(Agent):
    """
    An implementation of the Q Learning agent.

    """

    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1, beta=0.2):
        self.environment = env
        self.number_of_action = env.action_space.n
        self.q = defaultdict(lambda: np.zeros(self.number_of_action))
        self.r_avg = 0
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.policy = self._make_epsilon_greedy_policy()

    def _make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        """

        def policy_fn(state):
            A = np.ones(self.number_of_action, dtype=float) * self.epsilon / self.number_of_action
            best_action = np.argmax(self.q[state])
            A[best_action] += (1.0 - self.epsilon)
            return A

        return policy_fn

    def act(self, state):
        action_probs = self.policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q[next_state])
        td_target = reward + self.gamma * self.q[next_state][best_next_action]
        td_delta = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_delta

    def train(self, num_episodes=500, verbose=False):
        total_total_reward = 0.0
        rewards = []
        for i_episode in range(num_episodes):

            # Print out which episode we're on.
            if verbose:
                if (i_episode + 1) % 1 == 0:
                    print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                    sys.stdout.flush()

            state = self.environment.reset()
            state = str(state)
            total_reward = 0.0
            for t in itertools.count():
                action = self.act(state)
                next_state, reward, done, _ = self.environment.step(action)
                next_state = str(next_state)
                total_reward += reward

                self.update(state, action, reward, next_state)

                if done:
                    total_total_reward += total_reward
                    rewards.append(total_reward)
                    break

                state = next_state
        return total_total_reward / num_episodes, rewards  # return average eps reward
