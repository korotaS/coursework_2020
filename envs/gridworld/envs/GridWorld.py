from ast import literal_eval
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from collections import defaultdict
import matplotlib.pyplot as plt

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
GRAB = 4
ROTATE = 5
PUSH = 6
NUM_ACTIONS = 7

# Define colors
COLOURS = {0: [1, 1, 1], 1: [1, 0, 0], 2: [1, 1, 0.0],
           3: [1.0, 0.0, 1.0], 10: [0.0, 1, 0.0], 69: [0.0, 0.0, 1.0]}

MAP = "0 0 0 0 0 0 0 0 0 2 2 2 2\n" \
      "0 0 0 0 0 0 0 0 0 2 2 2 2\n" \
      "0 1 0 0 0 0 0 0 0 2 2 2 2\n" \
      "0 1 0 0 0 0 0 0 0 2 2 2 2\n" \
      "0 1 0 0 0 0 0 0 0 2 2 2 2\n" \
      "1 1 1 1 1 1 1 1 1 0 0 0 0\n" \
      "1 1 1 1 1 1 1 1 1 0 0 0 0\n" \
      "0 1 0 0 0 0 0 0 0 0 0 0 0\n" \
      "0 1 0 0 0 0 0 0 0 0 0 0 0\n" \
      "0 1 0 0 0 0 0 0 0 0 0 0 0\n" \
      "0 1 0 0 0 0 0 0 0 0 0 0 0\n" \
      "0 0 0 0 0 0 0 0 0 0 0 0 0\n" \
      "0 0 0 0 0 0 0 0 0 0 0 0 0"


class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, goal_reward=10.0, step_reward=-1.0, random_start_state=True, windiness=0.3):

        self.n = None
        self.m = None

        self.grid = None
        self.possibleStates = []

        self._map_init()

        self.random_start_state = random_start_state
        self.start_state_coord = (6, 0, 0)
        self.starting_state = self.start_state_coord
        self.state = self.start_state_coord
        self.door_grabbed = False
        self.door_active = False
        self.door_open = False
        self.policy_to_goal = []

        self.done = False

        self.goal = (5, 9)
        self.door_location = (5, 8)

        # Rewards
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.hit_red_reward = -30.0
        self.hit_yellow_reward = -15.0
        self.door_reward = 4

        # Gym spaces for observation and action space
        self.observation_space = spaces.Discrete(len(self.possibleStates))
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.windiness = windiness
        self.transition_probability = self._construct_transition_probability()

        self.np_random = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.rand(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if self.state[:2] == self.goal:
            self.done = True
            return self.state, self._get_reward(self.state), self.done, None
        reward = -2
        x, y, door = self.state
        if door != 1:
            self.door_grabbed = False
        curr_state = (x, y)
        if action == UP:
            x = x - 1
        elif action == DOWN:
            x = x + 1
        elif action == RIGHT:
            y = y + 1
        elif action == LEFT:
            y = y - 1
        elif action == GRAB and curr_state == self.door_location and \
                            not self.door_active and not self.door_open:
            self.door_grabbed = True
            reward = self.door_reward
            door = 1
        elif action == ROTATE and curr_state == self.door_location and \
                                 self.door_grabbed:
            self.door_active = True
            reward = self.door_reward
            door = 2
        elif action == PUSH and curr_state == self.door_location and \
                                 self.door_active:
            self.door_open = True
            reward = self.door_reward
            door = 3
        new_state = (x, y, door)

        if new_state[:2] == self.goal:
            if curr_state == self.door_location and self.door_open:
                self.state = new_state
        else:
            if self._get_grid_value(new_state) != -1:
                self.state = new_state

        # if self._get_grid_value(new_state) != -1:
        #     self.state = new_state

        if action <= 3:
            reward = self._get_reward(self.state)
        return self.state, reward, self.done, None

    def reset(self):
        self.done = False
        if self.random_start_state:
            start_points = [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0),
                            (6, 0, 0), (7, 0, 0), (8, 0, 0), (9, 0, 0), (10, 0, 0)]
            idx = np.random.randint(len(start_points))
            self.state = start_points[idx]  # self.start_state_coord
        else:
            self.state = self.start_state_coord
        self.starting_state = self.state
        self.door_open = False
        self.door_active = False
        return self.state

    def render(self, mode='human', draw_arrows=False, policy=None, name_prefix='FourRooms-v1 (G1)'):
        reward = self._build_policy_to_goal(policy)
        img = self._gridmap_to_img()
        fig = plt.figure(1, figsize=(10, 8), dpi=60,
                         facecolor='w', edgecolor='k')

        plt.clf()
        plt.xticks(np.arange(0, 14, 1))
        plt.yticks(np.arange(0, 14, 1))
        plt.grid(True)
        plt.title(name_prefix + "\nAgent:Purple, Goal:Green", fontsize=20)

        plt.imshow(img, origin="upper", extent=[0, 13, 0, 13])
        fig.canvas.draw()

        if draw_arrows & (type(policy) is not None):  # For drawing arrows of optimal policy
            fig = plt.gcf()
            ax = fig.gca()
            for state, action in policy.items():
                # y, x = literal_eval(state)[1]
                y, x, door = literal_eval(state)
                if door in [1, 2]:
                    continue
                y = 12 - y
                if action < 0:
                    pass
                    # style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
                    # if action == 4:
                    #     ax.text(x + 0.3, y + 0.3, 'O1', fontweight='bold')
                    # elif action == 5:
                    #     ax.text(x + 0.3, y + 0.3, 'O2', fontweight='bold')
                # elif action == ACTIVATE:
                #     ax.text(x + 0.3, y + 0.3, 'ACT', fontweight='bold')
                # elif action == PUSH:
                #     ax.text(x + 0.3, y + 0.3, 'PUS', fontweight='bold')
                elif action == -1:
                    ax.text(x + 0.3, y + 0.3, 'NE', fontweight='bold')
                else:
                    # self._draw_arrows(x, y, action)
                    pass

        plt.title(name_prefix + " learned Policy, reward: " + str(reward),
                  fontsize=15)

        plt.pause(0.00001)  # 0.01
        return

    def _build_policy_to_goal(self, policy):
        if type(policy) is not None:
            reward = 0
            curr_state = self.starting_state
            self.policy_to_goal = [curr_state]  # НЕ ЗАБЫТЬ ОТОБРАЗИТЬ ПУТЬ ДО ЦЕЛИ
            while curr_state[:2] != self.goal:
                reward += self._get_reward(curr_state)
                x, y, door = curr_state
                xa, ya, new_door = self._action_as_point(policy[str(curr_state)], door)
                curr_state = (x+xa, y+ya, new_door)
                self.policy_to_goal.append(curr_state)
            reward += self.door_reward
            reward += self.goal_reward
            print(self.policy_to_goal[-4:])
            return reward

    def _neighbouring(self, state, next_state):
        x1, y1 = state
        x2, y2 = next_state

        if x1 == x2 and y1 == y2:
            return True
        elif x1 == x2:
            if abs(y1 - y2) == 1:
                return True
            else:
                return False
        elif y1 == y2:
            if abs(x1 - x2) == 1:
                return True
            else:
                return False
        else:
            return False

    def _action_as_point(self, action, old_door):
        x = 0
        y = 0
        door = old_door
        if action == UP:
            x = x - 1
        elif action == DOWN:
            x = x + 1
        elif action == RIGHT:
            y = y + 1
        elif action == LEFT:
            y = y - 1
        elif action == GRAB:
            door = 1
        elif action == ROTATE:
            door = 2
        elif action == PUSH:
            door = 3
        elif action == NUM_ACTIONS:
            y = y + 1

        return x, y, door

    def _transition_probability(self, state, action, next_state):

        if not self._neighbouring(state, next_state):
            return 0.0

        x1, y1 = state
        xa, ya, _ = self._action_as_point(action, 0)
        x2, y2 = next_state

        if (x1 + xa == x2) and (y1 + ya == y2):
            return 1 - self.windiness + self.windiness / self.action_space.n
        elif state != next_state:
            return self.windiness / self.action_space.n
        # if (x1 + xa == x2) and (y1 + ya == y2):
        #     return 1
        # elif state != next_state:
        #     return 0

        return 0.0

    def _construct_transition_probability(self):
        p = defaultdict(lambda: [[] for _ in range(self.action_space.n)])
        for state in self.possibleStates:
            for action in range(self.action_space.n):
                pa = defaultdict(lambda: 0.0)
                for next_state in self.possibleStates:
                    pa[str(next_state)] = self._transition_probability(state, action, next_state)
                # for next_state in self.walls:
                #     pa[str(next_state)] = self._transition_probability(state, action, next_state)
                p[str(state)][action] = pa
        return p

    def _map_init(self):
        self.grid = []
        lines = MAP.split('\n')

        for i, row in enumerate(lines):
            row = row.split(' ')
            if self.n is not None and len(row) != self.n:
                raise ValueError(
                    "Map's rows are not of the same dimension...")
            self.n = len(row)
            rowArray = []
            for j, col in enumerate(row):
                rowArray.append(int(col))
                self.possibleStates.append((i, j))
            self.grid.append(rowArray)
        self.m = i + 1

    def _get_grid_value(self, state):
        if state[0] < 0 or state[0] > 12 or state[1] < 0 or state[1] > 12:
            return -1
        return self.grid[state[0]][state[1]]

    def _get_reward(self, state):
        if state[:2] == self.goal:
            return self.goal_reward
        elif self._get_grid_value(state) == 1:
            return self.hit_red_reward
        elif self._get_grid_value(state) == 2:
            return self.hit_yellow_reward
        else:
            return self.step_reward

    def _draw_arrows(self, x, y, direction):
        if direction == UP:
            x += 0.5
            dx = 0
            dy = 0.4
        if direction == DOWN:
            x += 0.5
            y += 1
            dx = 0
            dy = -0.4
        if direction == RIGHT:
            y += 0.5
            dx = 0.4
            dy = 0
        if direction == LEFT:
            x += 1
            y += 0.5
            dx = -0.4
            dy = 0

        plt.arrow(x,  # x1
                  y,  # y1
                  dx,  # x2 - x1
                  dy,  # y2 - y1
                  facecolor='k',
                  edgecolor='k',
                  width=0.005,
                  head_width=0.4,
                  )

    def _gridmap_to_img(self):
        row_size = len(self.grid)
        col_size = len(self.grid[0])

        obs_shape = [row_size, col_size, 3]

        img = np.zeros(obs_shape)

        gs0 = int(img.shape[0] / row_size)
        gs1 = int(img.shape[1] / col_size)
        for i in range(row_size):
            for j in range(col_size):
                for k in range(3):
                    if (i, j, 0) in self.policy_to_goal or (i, j, 1) in self.policy_to_goal:
                        this_value = COLOURS[69][k]
                    elif (i, j, 0) == self.state or (i, j, 1) == self.state:
                        this_value = COLOURS[10][k]
                    elif (i, j) == self.goal:
                        this_value = COLOURS[3][k]
                    else:

                        colour_number = int(self.grid[i][j])
                        this_value = COLOURS[colour_number][k]
                    img[i * gs0:(i + 1) * gs0, j * gs1:(j + 1)
                                                       * gs1, k] = this_value
        return img
