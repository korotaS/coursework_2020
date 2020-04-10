import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from agents.dqn.rewards import tolerance


DEGREES = 15
# ACTIONS = ['1CW', '1CCW', '2CW', '2CCW', '3CW', '3CCW', '4CW', '4CCW', 'grab', 'release']
ACTIONS = ['1CW', '1CCW', '2CW', '2CCW', '3CW', '3CCW', 'grab', 'release']
LENGTHS = [0, 1.5, 1.5]
BOUNDS = [[0, 359], [0, 150], [-120, 120]]
POSITIONS = [24, 11, 19]
TOL_BOUNDS = (0, 0.2)
TOL_MARGIN = 2
BLOCK_TO_ANGLE = {
    0: 315,
    1: 0,
    2: 45,
    3: 270,
    4: 90,
    5: 225,
    6: 180,
    7: 135
}
BLOCK_TO_COORDS = {
    0: [1.5, -1.5, 1],
    1: [1.5, 0, 1],
    2: [1.5, 1.5, 1],
    3: [0, -1.5, 1],
    4: [0, 1.5, 1],
    5: [-1.5, -1.5, 1],
    6: [-1.5, 0, 1],
    7: [-1.5, 1.5, 1]
}


class Manipulator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, situation, goal_reward=50.0, step_reward=-1.0,
                 windiness=0.3):
        self.state = None
        self.situation = situation
        self.num_of_joints = int((len(ACTIONS) - 2) / 2)
        self._map_init()
        self.done = False
        # Rewards
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.nice_action_reward = 50
        self.illegal_action_reward = -5
        self.observation_space = spaces.Discrete(self.num_of_joints+1)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.windiness = windiness
        self.np_random = None
        self.policy_to_goal = []

    def _map_init(self):
        self.manipulator_angles = self.situation['manipulator_angles']
        self.grabbed = self.situation['grabbed']
        # self.block_pos = self.situation['block_pos']
        self.block_pos = np.random.choice(list(range(8)))
        self.block_angle = BLOCK_TO_ANGLE[self.block_pos]
        self.block_coords = np.array(BLOCK_TO_COORDS[self.block_pos])
        self.task = self.situation['task']
        movement = np.array([0, 0, 1])
        if self.task == 'grab':
            self.goal_block_coords = self.block_coords + movement
            self.goal_man_coords = self.goal_block_coords
            self.goal_grabbed = True
        else:
            self.goal_block_coords = self.block_coords
            self.goal_man_coords = self.block_coords + movement
            self.goal_grabbed = False
        self.best_block_dist, self.best_man_dist = self._calculate_perfect_positions()
        self.state = self._encode(self.manipulator_angles, self.grabbed)
        self.starting_state = self.state

    def _calculate_obs_space_len(self):
        res = 1
        for pos in POSITIONS:
            res *= pos
        res *= 2
        return res

    def _calculate_perfect_positions(self):
        def lb(bound):
            while True:
                if bound % DEGREES == 0:
                    return bound
                bound += 1

        best_man_dist = np.inf
        best_block_dist = np.inf
        for first in range(lb(BOUNDS[0][0]), BOUNDS[0][1]+1, DEGREES):
            for second in range(lb(BOUNDS[1][0]), BOUNDS[1][1]+1, DEGREES):
                for third in range(lb(BOUNDS[2][0]), BOUNDS[2][1]+1, DEGREES):
                    if self.num_of_joints == 4:
                        for fourth in range(lb(BOUNDS[3][0]), BOUNDS[3][1]+1, DEGREES):
                            pos = self._calculate_hand_pos([first, second, third, fourth])
                            man_dist = self.distance(pos, self.goal_man_coords)
                            if man_dist < best_man_dist:
                                best_man_dist = man_dist

                            block_dist = self.distance(pos, self.block_coords)
                            if block_dist < best_block_dist:
                                best_block_dist = block_dist
                    else:
                        pos = self._calculate_hand_pos([first, second, third])
                        man_dist = self.distance(pos, self.goal_man_coords)
                        if man_dist < best_man_dist:
                            best_man_dist = man_dist

                        block_dist = self.distance(pos, self.block_coords)
                        if block_dist < best_block_dist:
                            best_block_dist = block_dist
        return best_block_dist, best_man_dist

    def _calculate_hand_pos(self, manipulator_angles):
        # without rotation
        def angle_n(n):
            return np.deg2rad(manipulator_angles[n])

        def rot_z_matr(angle):
            ct = np.cos(angle)
            st = np.sin(angle)
            return np.array([[ct, -st,  0],
                             [st, ct,  0],
                             [0, 0,  1]])
        temp_x = 0
        temp_z = 0
        temp_angle = 0
        for i in range(self.num_of_joints-1):
            temp_angle += angle_n(i+1)
            temp_x += LENGTHS[i + 1] * np.cos(temp_angle)
            temp_z += LENGTHS[i + 1] * np.sin(temp_angle)
        point = np.array([temp_x, 0, temp_z])

        # add rotation around z
        rot_matr = rot_z_matr(angle_n(0))
        point = rot_matr @ point
        point[2] += 1  # height of platform that holds the manipulator
        return point

    def _encode(self, manipulator_angles, grabbed):
        res = 0
        for i in range(self.num_of_joints-1):
            res += (manipulator_angles[i] - BOUNDS[i][0]) / DEGREES
            res *= POSITIONS[i]
        res += (manipulator_angles[-1] - BOUNDS[-1][0]) / DEGREES
        res = res * 2 + grabbed
        return res

    def _decode(self, i):
        grabbed = i % 2
        i //= 2
        manipulator_angles = []
        for j in range(self.num_of_joints-1):
            manipulator_angles.append(BOUNDS[-j-1][0] + (i % POSITIONS[-j-2])*DEGREES)
            i //= POSITIONS[-j-2]
        manipulator_angles.append(BOUNDS[0][0] + (i % POSITIONS[0]) * DEGREES)
        manipulator_angles.reverse()
        return manipulator_angles, grabbed

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.rand(seed)
        return [seed]

    def distance(self, point1, point2):
        return np.linalg.norm(point2 - point1)

    def reward(self, old_d, new_d):
        # return -5 if old_d - new_d <= 0 else 1
        return tolerance(new_d, bounds=TOL_BOUNDS, margin=TOL_MARGIN) / 10

    def normalize(self, manipulator_angles):
        normed = []
        for i, angle in enumerate(manipulator_angles):
            min_ = BOUNDS[i][0]
            max_ = BOUNDS[i][1]
            normed.append((angle-min_)/(max_-min_))
        return normed

    def denormalize(self, normed_man_angles):
        denormed = []
        for i, angle in enumerate(normed_man_angles):
            min_ = BOUNDS[i][0]
            max_ = BOUNDS[i][1]
            denormed.append(angle * (max_ - min_) + min_)
        return denormed

    def _norm_pos(self):
        return np.array([self.block_pos / 7])

    def step(self, action, return_all=False):
        assert self.action_space.contains(action)
        manipulator_angles, grabbed = self._decode(self.state)
        old_man_coords = self._calculate_hand_pos(manipulator_angles)
        old_distance = self.distance(old_man_coords, self.block_coords)
        if (self.task == 'grab' and grabbed or self.task == 'release' and not grabbed) and \
                self.distance(old_man_coords, self.goal_man_coords) < 0.2:
            self.done = True
            if return_all:
                return np.append(np.array(self.normalize(manipulator_angles) + [grabbed]), self._norm_pos()), \
                       self.goal_reward, self.done, None
            return self.state, self.goal_reward, self.done, None
        reward = 0
        if action < self.num_of_joints * 2:
            joint = int(action / 2)
            if action % 2 == 0:  # CW
                if manipulator_angles[joint] + DEGREES > BOUNDS[joint][1]:
                    reward = self.illegal_action_reward
                else:
                    manipulator_angles[joint] = manipulator_angles[joint] + DEGREES
                    new_man_pos = self._calculate_hand_pos(manipulator_angles)
                    new_distance = self.distance(new_man_pos, self.block_coords)
                    reward = self.reward(old_distance, new_distance)
            else:  # CCW
                if manipulator_angles[joint] - DEGREES < BOUNDS[joint][0]:
                    reward = self.illegal_action_reward
                else:
                    manipulator_angles[joint] = manipulator_angles[joint] - DEGREES
                    new_man_pos = self._calculate_hand_pos(manipulator_angles)
                    new_distance = self.distance(new_man_pos, self.block_coords)
                    reward = self.reward(old_distance, new_distance)
        elif action == self.num_of_joints * 2:  # grab
            if grabbed or self.task != 'grab' or old_distance > 0.2:
                reward = self.illegal_action_reward
            else:
                reward = self.nice_action_reward
                grabbed = True
                self.done = True
                if return_all:
                    return np.append(np.array(self.normalize(manipulator_angles) + [grabbed]), self._norm_pos()), \
                           self.goal_reward, self.done, None
                return self.state, self.goal_reward, self.done, None
        elif action == self.num_of_joints * 2 + 1:  # release
            if not grabbed or self.task != 'release' or old_distance > 0.2:
                reward = self.illegal_action_reward
            else:
                reward = self.nice_action_reward
                grabbed = False
                self.done = True
                if return_all:
                    return np.append(np.array(self.normalize(manipulator_angles) + [grabbed]), self._norm_pos()), \
                           self.goal_reward, self.done, None
                return self.state, self.goal_reward, self.done, None
        self.state = self._encode(manipulator_angles, grabbed)
        if return_all:
            return np.append(np.array(self.normalize(manipulator_angles) + [grabbed]), self._norm_pos()), \
                   reward, self.done, None
        return self.state, reward, self.done, None

    def reset(self, return_all=False):
        self.done = False
        self._map_init()
        if return_all:
            man_angles, grabbed = self._decode(self.state)
            return np.append(np.array(self.normalize(man_angles) + [grabbed]), self._norm_pos())
        return self.state

    def render(self, **kwargs):
        pass

    def build_policy_to_goal(self, policy, verbose=False, movement=False, save_path=None):
        if policy is None:
            raise AttributeError
        curr_state = self.starting_state
        count = 0
        done = False
        print()
        while not done:
            if count == 1000:
                print('some troubles with goal path...')
                return
            manipulator_angles, grabbed = self._decode(curr_state)
            print(manipulator_angles, grabbed)
            if self.task == 'grab' and grabbed or self.task == 'release' and not grabbed:
                done = True
            try:
                action = policy[str(curr_state)]
            except KeyError:
                print('some troubles with goal path...')
                return
            new_manipulator_angles, new_grabbed = self._action_as_point(action, manipulator_angles, grabbed)
            curr_state = self._encode(new_manipulator_angles, new_grabbed)
            self.policy_to_goal.append(curr_state)

    def _action_as_point(self, action, manipulator_angles, grabbed):
        new_grabbed = grabbed
        if action < self.num_of_joints * 2:
            joint = int(action / 2)
            if action % 2 == 0:  # CW
                manipulator_angles[joint] += DEGREES
            else:  # CCW
                manipulator_angles[joint] -= DEGREES
        elif action == self.num_of_joints * 2:  # grab
            new_grabbed = True
        elif action == self.num_of_joints * 2 + 1:  # release
            new_grabbed = False
        return manipulator_angles, new_grabbed
