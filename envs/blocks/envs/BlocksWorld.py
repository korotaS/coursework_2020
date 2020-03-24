import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt

COLOURS = {'empty': [1, 1, 1],
           'wall': [0, 0, 0],
           'block': [1, 0, 0],
           'path': [0, 1, 0],
           'path_block': [0, 0, 1]}
# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
PICKUP = 4
PUTDOWN = 5
NUM_ACTIONS = 6


class BlocksWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_dict, goal_reward=10.0, step_reward=-1.0,
                 windiness=0.3):
        self.num_cols = None
        self.num_rows = None
        self.block_start = None
        self.block_dest = None
        self.block_in_hand = None
        self.start_state_coord = None
        self.starting_state = None
        self.state = None
        self.goal = None
        self.grid = None
        self.walls = None
        self.possibleStates = []
        self.map_dict = map_dict
        self._map_init()
        self.policy_to_goal = []
        self.done = False
        # Rewards
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.nice_action_reward = 10
        self.illegal_action_reward = -10

        obs = (self.num_rows*self.num_cols)**3
        self.observation_space = spaces.Discrete(obs)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.windiness = windiness

        self.np_random = None

    def _map_init(self):
        self.num_rows = self.map_dict['map']['rows']
        self.num_cols = self.map_dict['map']['cols']
        if self.map_dict['map']['walls'] is not None:
            self.walls = []
            for wall in self.map_dict['map']['walls']:
                if wall[0] == wall[2]:  # horizontal wall
                    for y_c in range(wall[1], wall[3]+1):
                        self.walls.append([wall[0], y_c])
                if wall[1] == wall[3]:  # vertical wall
                    for x_c in range(wall[0], wall[2]+1):
                        if [x_c, wall[1]] not in self.walls:
                            self.walls.append([x_c, wall[1]])
        blocks = self.map_dict['blocks']
        block = blocks[0]
        if len(blocks) > 1:
            raise NotImplementedError
        self.block_in_hand = 0
        self.block_start = block['b_row'], block['b_col']
        self.block_dest = block['dest_row'], block['dest_col']
        start_coord = self.map_dict['start']['x'], self.map_dict['start']['y']
        goal_coord = self.map_dict['goal']['x'], self.map_dict['goal']['y']
        self.start_state_coord = start_coord
        self.starting_state = self._encode(start_coord, self.block_start, 0)
        self.state = self.starting_state
        self.goal = self._encode(goal_coord, self.block_dest, 0)

    def _encode_row_col(self, row_col):
        return row_col[0] * self.num_rows + row_col[1]

    def _decode_row_col(self, i):
        return i // self.num_rows, i % self.num_rows

    def _encode(self, agent, block, block_in_hand):
        rc = self.num_rows * self.num_cols
        return (self._encode_row_col(agent) * rc + self._encode_row_col(block))*2 + block_in_hand

    def _decode(self, i):
        rc = self.num_rows * self.num_cols
        out = [i % 2]
        i = i // 2
        out.append(i % rc)
        i = i // rc
        out.append(i)
        return self._decode_row_col(out[2]), self._decode_row_col(out[1]), out[0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.rand(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if self.state == self.goal:
            self.done = True
            return self.state, self.goal_reward, self.done, None
        (a_x, a_y), (b_x, b_y), b_in_h = self._decode(self.state)
        reward = -1
        if action == UP:
            a_x = a_x - 1
            if b_in_h == 1:
                b_x = b_x - 1
        elif action == DOWN:
            a_x = a_x + 1
            if b_in_h == 1:
                b_x = b_x + 1
        elif action == RIGHT:
            a_y = a_y + 1
            if b_in_h == 1:
                b_y = b_y + 1
        elif action == LEFT:
            a_y = a_y - 1
            if b_in_h == 1:
                b_y = b_y - 1
        elif action == PICKUP:
            if b_in_h == 0 and (a_x, a_y) == (b_x, b_y) and (a_x, a_y) != self.block_dest:
                b_in_h = 1
                reward = self.nice_action_reward
            else:
                reward = self.illegal_action_reward
        elif action == PUTDOWN:
            if b_in_h == 1 and (a_x, a_y) == self.block_dest:
                b_in_h = 0
                reward = self.nice_action_reward
            else:
                reward = self.illegal_action_reward
        new_state = self._encode((a_x, a_y), (b_x, b_y), b_in_h)
        if self._is_possible_move(new_state):
            self.state = new_state
        return self.state, reward, self.done, None

    def _is_possible_move(self, state):
        (a_x, a_y), _, _ = self._decode(state)
        walls_check = True
        if self.walls is not None:
            walls_check = [a_x, a_y] not in self.walls
        return 0 < a_x < self.num_rows and 0 < a_y < self.num_cols and walls_check

    def reset(self):
        self.done = False
        self._map_init()
        return self.state

    def render(self, policy=None, name_prefix='BlocksWorld'):
        self.build_policy_to_goal(policy)
        img = self._map_to_img()
        fig = plt.figure(1, figsize=(10, 8), dpi=60,
                         facecolor='w', edgecolor='k')
        plt.clf()
        plt.xticks(np.arange(0, self.num_cols+1, 1))
        plt.yticks(np.arange(self.num_rows+1, 0, -1))
        plt.grid(True)
        plt.title(name_prefix + "\nAgent:Purple, Goal:Green", fontsize=20)
        plt.imshow(img, origin="upper", extent=[0, self.num_rows, 0, self.num_cols])
        fig.canvas.draw()
        plt.title(name_prefix + " learned Policy", fontsize=15)

        plt.pause(0.00001)  # 0.01
        return

    def _map_to_img(self):
        img = np.zeros((self.num_rows, self.num_cols, 3))
        gs0 = int(img.shape[0] / self.num_rows)
        gs1 = int(img.shape[1] / self.num_cols)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                for k in range(3):
                    if self.walls is not None and [i, j] in self.walls:
                        this_value = COLOURS['wall'][k]
                    elif (i, j) == self.block_start or (i, j) == self.block_dest:
                        this_value = COLOURS['block'][k]
                    else:
                        this_value = COLOURS['empty'][k]
                    img[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1, k] = this_value
        for step in self.policy_to_goal:
            (xa, ya), _, b_i_h = self._decode(step)
            if b_i_h:
                this_value = COLOURS['path_block']
            else:
                this_value = COLOURS['path']
            img[xa * gs0:(xa + 1) * gs0, ya * gs1:(ya + 1) * gs1, :] = this_value
        return img

    def build_policy_to_goal(self, policy):
        if type(policy) is not None:
            curr_state = self.starting_state
            self.policy_to_goal = []
            while curr_state != self.goal:
                (xa, ya), (xb, yb), b_i_h = self._decode(curr_state)
                print(f'agent: {xa, ya}; block: {xb, yb}; in hand: {b_i_h}')
                (dxa, dya), (dxb, dyb), db_in_h = self._action_as_point(policy[str(curr_state)], b_i_h)
                curr_state = self._encode((xa+dxa, ya+dya),
                                          (xb+dxb, yb+dyb),
                                          db_in_h)
                self.policy_to_goal.append(curr_state)
            print(f'agent: {xa+dxa, ya+dya}; block: {xb+dxb, yb+dyb}; in hand: {db_in_h}')

    def _action_as_point(self, action, b_in_h):
        a_x = 0
        a_y = 0
        b_x = 0
        b_y = 0
        if action == UP:
            a_x = a_x - 1
            if b_in_h == 1:
                b_x = b_x - 1
        elif action == DOWN:
            a_x = a_x + 1
            if b_in_h == 1:
                b_x = b_x + 1
        elif action == RIGHT:
            a_y = a_y + 1
            if b_in_h == 1:
                b_y = b_y + 1
        elif action == LEFT:
            a_y = a_y - 1
            if b_in_h == 1:
                b_y = b_y - 1
        elif action == PICKUP:
            if b_in_h == 0:
                b_in_h = 1
        elif action == PUTDOWN:
            if b_in_h == 1:
                b_in_h = 0
        return (a_x, a_y), (b_x, b_y), b_in_h
