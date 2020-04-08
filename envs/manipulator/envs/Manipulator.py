import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


DEGREES = 15
# ACTIONS = ['1CW', '1CCW', '2CW', '2CCW', '3CW', '3CCW', '4CW', '4CCW', 'grab', 'release']
ACTIONS = ['1CW', '1CCW', '2CW', '2CCW', '3CW', '3CCW', 'grab', 'release']
LENGTHS = [0, 2, 1.5]
BOUNDS = [[0, 359], [-75, 75], [-135, 135]]
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

    def __init__(self, situation, goal_reward=10.0, step_reward=-1.0,
                 windiness=0.3):
        self.state = None
        self.situation = situation
        self.num_of_positions = (360 / DEGREES)
        self.num_of_joints = int((len(ACTIONS) - 2) / 2)
        self._map_init()
        self.done = False
        # Rewards
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.nice_action_reward = 5
        self.illegal_action_reward = -5
        self.observation_space = spaces.Discrete(self.num_of_positions**self.num_of_joints * 2 * 8)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.windiness = windiness
        self.np_random = None
        self.policy_to_goal = []

    def _map_init(self):
        self.manipulator_angles = self.situation['manipulator_angles']
        self.grabbed = self.situation['grabbed']
        self.block_pos = self.situation['block_pos']
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
        self.state = self._encode(self.manipulator_angles, self.grabbed, self.block_pos)
        self.starting_state = self.state

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

    def _encode(self, manipulator_angles, grabbed, block_pos):
        res = 0
        angles_minus = [1 if angle < 0 else 0 for angle in manipulator_angles]
        for i in range(self.num_of_joints-1):
            res += abs(manipulator_angles[i] / DEGREES)
            res *= self.num_of_positions
        res += abs(manipulator_angles[-1] / DEGREES)

        res *= 2

        for i in range(self.num_of_joints-1):
            res += angles_minus[i]
            res *= 2
        res += angles_minus[-1]

        res = res * 2 + grabbed
        res = res * 8 + block_pos
        return res

    def _decode(self, i):
        block_pos = i % 8
        i //= 8
        grabbed = i % 2
        i //= 2

        angles_minus = []
        for _ in range(self.num_of_joints):
            angles_minus.append(i % 2)
            i //= 2
        angles_minus.reverse()

        manipulator_angles = []
        for _ in range(self.num_of_joints):
            manipulator_angles.append((i % self.num_of_positions)*DEGREES)
            i //= self.num_of_positions
        manipulator_angles.reverse()

        manipulator_angles = [-angle if angles_minus[i] == 1 else angle
                              for i, angle in enumerate(manipulator_angles)]
        return manipulator_angles, grabbed, block_pos

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.rand(seed)
        return [seed]

    def distance(self, point1, point2):
        return np.linalg.norm(point2 - point1)

    def reward(self, old_d, new_d):
        return -5 if old_d - new_d <= 0 else 1

    def step(self, action):
        assert self.action_space.contains(action)
        manipulator_angles, grabbed, block_pos = self._decode(self.state)
        old_man_coords = self._calculate_hand_pos(manipulator_angles)
        old_distance = self.distance(old_man_coords, self.block_coords)
        if (self.task == 'grab' and grabbed or self.task == 'release' and not grabbed) and \
                self.distance(old_man_coords, self.goal_man_coords) < 0.25:
            self.done = True
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
            if grabbed or self.task != 'grab' or old_distance != self.best_block_dist:
                reward = self.illegal_action_reward
            else:
                grabbed = True
                reward = self.nice_action_reward
        elif action == self.num_of_joints * 2 + 1:  # release
            if not grabbed or self.task != 'release' or old_distance != self.best_block_dist:
                reward = self.illegal_action_reward
            else:
                grabbed = False
                reward = self.nice_action_reward
        self.state = self._encode(manipulator_angles, grabbed, block_pos)
        return self.state, reward, self.done, None

    def reset(self):
        self.done = False
        self._map_init()
        return self.state

    def render(self, **kwargs):
        pass

    # def _action_to_str(self, action):
    #     if action == UP:
    #         return 'up'
    #     elif action == DOWN:
    #         return 'down'
    #     elif action == RIGHT:
    #         return 'right'
    #     elif action == LEFT:
    #         return 'left'
    #     elif action == PICKUP:
    #         return 'pick up'
    #     elif action == PUTDOWN:
    #         return 'put down'
    #
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
            manipulator_angles, grabbed, block_pos = self._decode(curr_state)
            print(manipulator_angles, grabbed)
            if self.task == 'grab' and grabbed or self.task == 'release' and not grabbed:
                done = True
            try:
                action = policy[str(curr_state)]
            except KeyError:
                print('some troubles with goal path...')
                return
            new_manipulator_angles, new_grabbed = self._action_as_point(action, manipulator_angles, grabbed)
            curr_state = self._encode(new_manipulator_angles, new_grabbed, block_pos)
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
