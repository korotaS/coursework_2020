import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import os
import time
import imageio

COLOURS = {'empty': [1, 1, 1],
           'wall': [0, 0, 0],
           'path': [0, 1, 0],
           'path_block': [0, 0, 1],
           'block': [1, 0, 0]}
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
        self.walls = None
        self.state = None
        self.possibleStates = []
        self.map_dict = map_dict
        self._map_init()
        self.policy_to_goal = []
        self.done = False
        # Rewards
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.nice_action_reward = 5
        self.illegal_action_reward = -5

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
        self.blocks_start = {}
        self.blocks_dest = {}
        self.map_start_x = self.map_dict['map']['start_x']
        self.map_start_y = self.map_dict['map']['start_y']
        self.coord_modes = []
        self.block_names = []
        self.delivered = []
        for name, block in blocks.items():
            if block['start_x'] != block['goal_x'] and block['start_x'] == block['goal_x']:
                self.blocks_start[name] = {'x': block['start_x'], 'y': block['start_y']}
                self.blocks_dest[name] = {'x': block['goal_x'], 'y': block['goal_y']}
                self.delivered.append(False)
                self.coord_modes.append(block['coord_mode'])
                self.block_names.append(name)
        self.num_blocks = len(self.delivered)
        agent = self.map_dict['agent']
        self.agent_coord_mode = agent['coord_mode']
        b_in_h = 0
        if agent['holding_start'] is not None:
            b_in_h = list(self.blocks_start.keys()).index(agent['holding_start'])+1
        start_coord = agent['start_x'], agent['start_y']
        goal_coord = agent['goal_x'], agent['goal_y']
        self.b_in_h_goal = 0
        for i, val in enumerate(self.blocks_dest.values()):
            if (val['x'], val['y']) == goal_coord:
                self.b_in_h_goal = i + 1
                break
        self.start_coord = start_coord
        self.goal_coord = goal_coord
        self.starting_state = self._encode(start_coord, self.blocks_start, b_in_h)
        self.state = self.starting_state
        self.goal = self._encode(goal_coord, self.blocks_dest, self.b_in_h_goal)

    def _encode_row_col(self, row_col):
        return row_col[0] * self.num_rows + row_col[1]

    def _decode_row_col(self, i):
        return [i // self.num_rows, i % self.num_rows]

    def _encode(self, agent, blocks, block_in_hand):
        rc = self.num_rows * self.num_cols
        i = self._encode_row_col(agent)
        i *= rc
        for j, (_, v) in enumerate(blocks.items()):
            block_info = [v['x'], v['y']]
            i += self._encode_row_col(block_info)
            if j != self.num_blocks-1:
                i *= rc
            else:
                i *= (self.num_blocks+1)
                i += block_in_hand
        return i

    def _decode(self, i):
        rc = self.num_rows * self.num_cols
        b_in_h = i % (self.num_blocks+1)
        i = i // (self.num_blocks+1)
        blocks = {}
        for j in range(self.num_blocks):
            x, y = self._decode_row_col(i % rc)
            blocks[j] = {'x': x, 'y': y}
            i = i // rc
        if self.num_blocks == 0:
            i = i // rc
        blocks = {k: blocks[k] for k in reversed(list(blocks.keys()))}
        agent = self._decode_row_col(i)
        return agent, blocks, b_in_h

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.rand(seed)
        return [seed]

    def _is_near_block(self, a_x, a_y, block):
        return block == [a_x - 1, a_y - 1] or \
               block == [a_x - 1, a_y] or \
               block == [a_x - 1, a_y + 1] or \
               block == [a_x, a_y - 1] or \
               block == [a_x, a_y + 1] or \
               block == [a_x + 1, a_y - 1] or \
               block == [a_x + 1, a_y] or \
               block == [a_x + 1, a_y + 1]

    def _blocks_dict_to_arr(self, blocks):
        return [[val['x'], val['y']] for val in blocks.values()]

    def _blocks_arr_to_dict(self, blocks_arr):
        blocks_dict = {i: {} for i in range(len(blocks_arr))}
        for i, block in enumerate(blocks_arr):
            blocks_dict[i]['x'] = blocks_arr[i][0]
            blocks_dict[i]['y'] = blocks_arr[i][1]
        return blocks_dict

    def step(self, action):
        assert self.action_space.contains(action)
        if self.state == self.goal:
            self.done = True
            return self.state, self.goal_reward, self.done, None
        (a_x, a_y), blocks, b_in_h = self._decode(self.state)
        blocks_arr = self._blocks_dict_to_arr(blocks)
        reward = -1
        if b_in_h == 0:
            try:
                curr_block_index = self.delivered.index(False)
            except ValueError:
                curr_block_index = -1
        else:
            curr_block_index = b_in_h - 1
        if action == UP and a_x > 0:
            a_x = a_x - 1
            if b_in_h != 0:
                blocks_arr[b_in_h-1][0] -= 1
        elif action == DOWN and a_x < self.num_rows - 1:
            a_x = a_x + 1
            if b_in_h != 0:
                blocks_arr[b_in_h - 1][0] += 1
        elif action == RIGHT and a_y < self.num_cols - 1:
            a_y = a_y + 1
            if b_in_h != 0:
                blocks_arr[b_in_h - 1][1] += 1
        elif action == LEFT and a_y > 0:
            a_y = a_y - 1
            if b_in_h != 0:
                blocks_arr[b_in_h - 1][1] -= 1
        elif curr_block_index == -1:
            reward = self.illegal_action_reward
        else:
            if action == PICKUP:
                if b_in_h == 0 and self._is_near_block(a_x, a_y, blocks_arr[curr_block_index]):
                    blocks_arr[curr_block_index] = [a_x, a_y]
                    b_in_h = curr_block_index + 1
                    reward = self.nice_action_reward
                else:
                    reward = self.illegal_action_reward
            elif action == PUTDOWN:
                curr_block_dest = None
                if b_in_h > 0 and b_in_h != self.b_in_h_goal:
                    key = list(self.blocks_dest.keys())[curr_block_index]
                    curr_block_dest = [self.blocks_dest[key]['x'], self.blocks_dest[key]['y']]
                if b_in_h != 0 and self._is_near_block(a_x, a_y, curr_block_dest):
                    b_in_h = 0
                    blocks_arr[curr_block_index] = curr_block_dest
                    reward = self.nice_action_reward
                    self.delivered[curr_block_index] = True
                else:
                    reward = self.illegal_action_reward
        blocks = self._blocks_arr_to_dict(blocks_arr)
        new_state = self._encode((a_x, a_y), blocks, b_in_h)
        if self._is_possible_move(new_state):
            self.state = new_state
        return self.state, reward, self.done, None

    def _is_possible_move(self, state):
        (a_x, a_y), _, _ = self._decode(state)
        if self.walls is not None:
            if [a_x, a_y] in self.walls:
                return False
        blocks_start_arr = [[block['x'], block['y']]
                            for block in self.blocks_start.values()]
        blocks_dest_arr = [[block['x'], block['y']]
                           for block in self.blocks_dest.values()]
        blocks_on_ground = [blocks_dest_arr[i] if val
                            else blocks_start_arr[i]
                            for i, val in enumerate(self.delivered)]
        if [a_x, a_y] in blocks_on_ground:
            return False
        return True

    def reset(self):
        self.done = False
        self._map_init()
        return self.state

    def _action_to_str(self, action):
        if action == UP:
            return 'up'
        elif action == DOWN:
            return 'down'
        elif action == RIGHT:
            return 'right'
        elif action == LEFT:
            return 'left'
        elif action == PICKUP:
            return 'pick up'
        elif action == PUTDOWN:
            return 'put down'

    def build_policy_to_goal(self, policy, verbose=False, movement=False):
        if type(policy) is not None:
            curr_state = self.starting_state
            self.policy_to_goal = []
            delivered = [False for _ in range(self.num_blocks)]
            for i, name in enumerate(self.blocks_dest):
                if self.blocks_dest[name] == self.blocks_start[name]:
                    delivered[i] = True
            count = 0
            full_path = ''
            print('\nCalculating full path...')
            if movement:
                dir_name = time.strftime('%d_%b_%Y_%H_%M_%S')
                full_path = f'movements/{dir_name}/'
                os.mkdir(full_path)
            while curr_state != self.goal:
                (xa, ya), blocks, b_i_h = self._decode(curr_state)
                try:
                    action = policy[str(curr_state)]
                except KeyError:
                    print('some troubles with goal path...')
                    return
                if movement:
                    self.render([curr_state], save_path=full_path+f'{count}.png',
                                name_prefix=f'Agent at {count}th time step')
                count += 1
                if verbose:
                    string = f'agent: {xa, ya}; '
                    if self.agent_coord_mode == 'cropped':
                        string = f'agent: {xa+self.map_start_x, ya+self.map_start_y}; '
                    for i, (_, v) in enumerate(blocks.items()):
                        x, y = v["x"], v["y"]
                        if self.coord_modes[i] == 'cropped':
                            x += self.map_start_x
                            y += self.map_start_y
                        string += f'{self.block_names[i]}: {x, y}; '
                    print(string + f'in hand: ' +
                          (f'block_{b_i_h}; ' if b_i_h != 0 else 'nothing; ') +
                          f'action: {self._action_to_str(action)}')
                (dxa, dya), new_blocks, db_in_h, delivered = self._action_as_point(action,
                                                                                   blocks,
                                                                                   b_i_h,
                                                                                   [xa, ya],
                                                                                   delivered)
                curr_state = self._encode((xa+dxa, ya+dya),
                                          new_blocks,
                                          db_in_h)
                self.policy_to_goal.append(curr_state)
            if movement:
                print('Making gif...')
                images = []
                gif_path = full_path + 'movement.gif'
                if os.path.exists(gif_path):
                    os.remove(gif_path)
                one_frame_duration = 0.5
                kwargs = {'duration': one_frame_duration}
                for filename in sorted(os.listdir(full_path), key=lambda file: int(file.split('.')[0])):
                    images.append(imageio.imread(full_path + filename))
                imageio.mimsave(gif_path, images, 'GIF', **kwargs)

    def _action_as_point(self, action, blocks, b_in_h, old_coords, delivered):
        a_x = 0
        a_y = 0
        blocks_arr = self._blocks_dict_to_arr(blocks)
        if b_in_h == 0:
            try:
                curr_block_index = delivered.index(False)
            except ValueError:
                curr_block_index = -1
        else:
            curr_block_index = b_in_h - 1
        if action == UP:
            a_x = a_x - 1
            if b_in_h != 0:
                blocks_arr[b_in_h-1][0] -= 1
        elif action == DOWN:
            a_x = a_x + 1
            if b_in_h != 0:
                blocks_arr[b_in_h - 1][0] += 1
        elif action == RIGHT:
            a_y = a_y + 1
            if b_in_h != 0:
                blocks_arr[b_in_h - 1][1] += 1
        elif action == LEFT:
            a_y = a_y - 1
            if b_in_h != 0:
                blocks_arr[b_in_h - 1][1] -= 1
        elif action == PICKUP:
            if b_in_h == 0:
                b_in_h = curr_block_index + 1
                blocks_arr[curr_block_index] = old_coords
        elif action == PUTDOWN:
            if b_in_h != 0:
                b_in_h = 0
                delivered[curr_block_index] = True
                key = list(self.blocks_dest.keys())[curr_block_index]
                curr_block = [self.blocks_dest[key]['x'], self.blocks_dest[key]['y']]
                blocks_arr[curr_block_index] = curr_block
        blocks = self._blocks_arr_to_dict(blocks_arr)
        return (a_x, a_y), blocks, b_in_h, delivered

    # drawing and animation methods

    def render(self, policy=None, name_prefix='BlocksWorld', save_path=None):
        if save_path is not None:
            img = self._map_to_img(policy)
        else:
            self.build_policy_to_goal(policy, verbose=False)
            img = self._map_to_img(self.policy_to_goal)
        fig = plt.figure(1, figsize=(10, 8), dpi=50,
                         facecolor='w', edgecolor='k')
        plt.clf()
        plt.xticks(np.arange(0, self.num_cols+1, 1))
        plt.yticks(np.arange(0, self.num_rows+1, 1))
        plt.grid(True)
        plt.imshow(img, origin="lower", extent=[0, self.num_cols, 0, self.num_rows])
        plt.title(name_prefix, fontsize=15)
        plt.gca().invert_yaxis()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            fig.canvas.draw()
            plt.pause(0.00001)
        return

    def _map_to_img(self, policy):
        img = np.zeros((self.num_rows, self.num_cols, 3))
        gs0 = int(img.shape[0] / self.num_rows)
        gs1 = int(img.shape[1] / self.num_cols)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                for k in range(3):
                    if self.walls is not None and [i, j] in self.walls:
                        this_value = COLOURS['wall'][k]
                    else:
                        this_value = COLOURS['empty'][k]
                    img[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1, k] = this_value
        for step in policy:
            (xa, ya), blocks, b_i_h = self._decode(step)
            if b_i_h:
                this_value = COLOURS['path_block']
            else:
                this_value = COLOURS['path']
            img[xa * gs0:(xa + 1) * gs0, ya * gs1:(ya + 1) * gs1, :] = this_value
        if len(policy) == 1:
            _, blocks, b_i_h = self._decode(policy[0])
            for i, block in enumerate(blocks):
                x, y = block
                if i != b_i_h - 1:
                    img[x * gs0:(x + 1) * gs0, y * gs1:(y + 1) * gs1, :] = COLOURS['block']
        return img
