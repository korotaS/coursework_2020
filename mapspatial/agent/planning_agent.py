import importlib
import logging
import math
import sys
import time
import os
from copy import deepcopy, copy
from multiprocessing import Process, Pipe
import platform

sys.setrecursionlimit(2500)

from mapspatial.grounding import json_grounding
from mapspatial.grounding.utils import signs_markup, state_prediction, define_situation, define_map, \
    state_fixation, locater, cell_creater
from mapspatial.search.mapsearch import SpSearch
from mapcore.planning.agent.planning_agent import PlanningAgent

SIT_SUF = 0

if platform.system() != 'Windows':
    delim = '/'
else:
    delim = '\\'
load_dir = os.getcwd() + delim + 'mapspatial' + delim + 'temp' + delim

class SpAgent(PlanningAgent):
    def __init__(self):
        super().__init__()

    # Initialization
    def initialize(self, name, agents, problem, backward, subsearch):
        """
        This function allows agent to be initialized. We do not use basic __init__ to let
        user choose a valid variant of agent. You can take agent with othe abilities.
        :param problem: problem
        :param ref: the dynamic value of plan clarification
        """
        super().initialize(problem, 'spatial', backward)
        self.name = name
        self.agents = agents
        self.problem = problem
        self.solution = []
        self.final_solution = ''
        self.backward = backward
        self.task_file = problem.task_file
        self.subsearch = subsearch
        self.task = None


    # Grounding tasks
    def get_task(self):
        """
        This functions is needed to update current agents' SWM
        :return: task - sign representation of the problem.
        """
        logging.info('Начато означивание: {0}'.format(self.problem.name))
        signs = self.load_swm(type = 'spatial')
        self.task = json_grounding.spatial_ground(self.problem, self.name, self.agents, signs, self.backward)
        logging.info('Означивание окончено: {0}'.format(self.problem.name))
        logging.info('{0} знаков найдено'.format(len(self.task.signs)))
        if signs:
            return self.task, len(signs) - len(self.task.signs)
        else:
            return self.task, 0

    def get_scenario(self, problem_file, benchmark):
        """
        This function call multiMAP if agent is not alone or mapcore in other side.
        After synthesis pddl plan function create a spatial representation of plan
        situations.
        :param problem_file:
        :param benchmark:
        :return: tuple(pddl action, spatial effect situation, spatial effect map)
        """
        if platform.system() != 'Windows':
            delim = '/'
            projectPath = delim+''.join([el+delim for el in os.getcwd().split(delim)[1:-1]])
        else:
            delim = '\\'
            projectPath = ''.join([el+delim for el in os.getcwd().split(delim)[:-1]])
        if len(self.agents) > 1:
            sys.path.append(projectPath+'map-multi')
            import test1
            main = getattr(test1, 'main')
            pddl_solutions = main([problem_file, "mapmulti.agent.planning_agent", "MAgent", "False"])
        else:
            sys.path.append(projectPath+'map-core')
            from mapcore import test0
            main = getattr(test0, 'main')
            pddl_solutions = main([problem_file, "mapcore.planning.agent.planning_agent", "PlanningAgent", "False"])

        pddl_solution = pddl_solutions[self.name][0][0]
        #sol_goal = pddl_solutions[self.name][0][1]
        path_to_file = pddl_solutions[self.name][1]
        pddl_signs = self.load_swm(path = path_to_file)

        scenario = []
        cl_lv = 0
        size = None
        prev_ag = ''
        if benchmark == 'blocks':
            block_signs = set()
            for _, cm in pddl_signs['block'].significances.items():
                block_signs |= cm.get_signs()
            cur_sit = self.task.additions[0][max(self.task.additions[0])]
            for action in pddl_solution:
                if prev_ag != action[3].name:
                    size = None
                    cl_lv = 0
                spat_cm, spat_map, cl_lv, new_sit, size = self.get_spatial_sit_blocks(action, cur_sit, size, cl_lv)
                scenario.append(((action[1], action[3].name), spat_cm, spat_map, cur_sit, new_sit, cl_lv))
                cur_sit = new_sit
                prev_ag = action[3].name
        return scenario

    def get_spatial_sit_blocks(self, action, cur_sit, prev_size, prev_cl):
        if action[3].name == 'I':
            ag_name = self.name
        else:
            ag_name = action[3].name
        new_sit = deepcopy(cur_sit)
        ag_pos = new_sit['objects'][ag_name]
        ag_activity = new_sit[ag_name]['activity']
        ground_block = None
        clear_block = None
        move_block = None

        # we need to know coords of block in effect
        for el in action[2].effect:
            el_signs = el.get_signs()
            if (action[1] == 'pick-up' or action[1] == 'unstack')\
                    and 'holding' in [sign.name for sign in el_signs]:
                move_block = [sign for sign in el_signs if sign.name != 'holding' and sign.name != ag_name][0]
                break
            elif (action[1] == 'stack' or action[1] == 'put-down') and 'clear' in [sign.name for sign in el_signs]:
                clear_block = [sign for sign in el_signs if sign.name != 'clear'][0]
                break
        if action[1] == 'stack':
            for el in action[2].cause:
                el_signs = el.get_signs()
                if 'clear' in [sign.name for sign in el_signs]:
                    ground_block = [sign for sign in el_signs if sign.name != 'clear'][0]
                    break
        if action[1] == 'unstack':
            for el in action[2].effect:
                el_signs = el.get_signs()
                if 'clear' in [sign.name for sign in el_signs]:
                    ground_block = [sign for sign in el_signs if sign.name != 'clear'][0]
                    break


        if action[1] == 'put-down' or action[1] == 'stack':
            obj = clear_block
        else:
            obj = move_block

        def get_pos(ag_pos, obj_pos):
            """
            Thus function was implemented to foresee the agent's position
            after it will manipulate with object.
            """
            if ag_pos['x'] < obj_pos['x']:
                if ag_pos['y'] < obj_pos['y']-ag_pos['r']:
                    x = obj_pos['x']- obj_pos['r'] - 4*ag_pos['r']
                    y = obj_pos['y'] - obj_pos['r'] - 4*ag_pos['r']
                    orient = 'below-right'
                elif ag_pos['y'] > obj_pos['y']+ag_pos['r']:
                    x = obj_pos['x']- obj_pos['r'] - 10*ag_pos['r']
                    y = obj_pos['y'] + obj_pos['r']+ 10*ag_pos['r']
                    orient = 'above-right'
                else:
                    x = obj_pos['x'] - obj_pos['r'] - 4*ag_pos['r']
                    y = obj_pos['y']
                    orient = 'right'
            elif ag_pos['x'] > obj_pos['x']:
                if ag_pos['y'] < obj_pos['y']-ag_pos['r']:
                    x = obj_pos['x'] + obj_pos['r'] + 4*ag_pos['r']
                    y = obj_pos['y'] - obj_pos['r'] - 4*ag_pos['r']
                    orient = 'below-left'
                elif ag_pos['y'] > obj_pos['y']+ag_pos['r']:
                    x = obj_pos['x'] + obj_pos['r'] + 4*ag_pos['r']
                    y = obj_pos['y'] + obj_pos['r'] + 4*ag_pos['r']
                    orient = 'above-left'
                else:
                    x = obj_pos['x'] + obj_pos['r'] + 2*ag_pos['r']
                    y = obj_pos['y']
                    orient = 'left'
            else:
                if ag_pos['y'] < obj_pos['y']:
                    x = ag_pos['x']
                    y = obj_pos['y'] - obj_pos['r'] - 2*ag_pos['r']
                    orient = 'below'
                else:
                    x = ag_pos['x']
                    y = obj_pos['y'] + obj_pos['r'] + 2*ag_pos['r']
                    orient = 'above'
            return x, y, orient

        # todo change to set of applicable poses
        size = None
        cl_lv = 0
        if action[1] == 'pick-up' or action[1] == 'unstack':
            obj_pos = new_sit['objects'][obj.name]
            x, y, orient = get_pos(ag_pos, obj_pos)
            ag_pos['x'] = x
            ag_pos['y'] = y
            # check old size, on which agent can perform an action
            _, _, _, _, _, size, cl_lv = signs_markup(new_sit, self.task.additions[3],ag_name)
            new_sit['objects'].pop(obj.name)
            for pred, signif in new_sit[ag_name].copy().items():
                if pred == 'orientation':
                    new_sit[ag_name].pop(pred)
                    new_sit[ag_name][pred] = orient
                elif pred == 'handempty':
                    new_sit[ag_name].pop(pred)
                    new_sit[ag_name]['holding'] = {'cause':[ag_name, move_block.name], 'effect':[]}
        elif action[1] == 'put-down':
            new_sit['objects'][obj.name] = {}
            obj_pos = new_sit['objects'][obj.name]
            obj_pos['x'] = ag_pos['x']+ 2*ag_pos['r'] + ag_activity // 2
            obj_pos['y'] = ag_pos['y']+ 2*ag_pos['r'] + ag_activity // 2
            obj_pos['r'] = self.task.additions[0][0]['objects'][obj.name]['r']
        elif action[1] == 'stack':
            ground_coords = new_sit['objects'][ground_block.name]
            x, y, orient = get_pos(ag_pos, ground_coords)
            # if we need to move, or stay on prev position
            if math.sqrt((ag_pos['y']-y)**2 + (ag_pos['x']-x)**2) >= ag_pos['r']*2:
                ag_pos['x'] = x
                ag_pos['y'] = y
            new_sit['objects'][obj.name] = {}
            obj_pos = new_sit['objects'][obj.name]
            obj_pos['x'] = ground_coords['x']
            obj_pos['y'] = ground_coords['y']
            obj_pos['r'] = self.task.additions[0][0]['objects'][obj.name]['r']
            for pred, signif in new_sit[ag_name].copy().items():
                if pred == 'orientation':
                    new_sit[ag_name].pop(pred)
                    new_sit[ag_name][pred] = orient
                elif pred == 'holding':
                    new_sit[ag_name].pop(pred)
                    new_sit[ag_name]['handempty'] = {'cause':[], 'effect':[]}
        region_map, cell_map_pddl, cell_location, near_loc, cell_coords, size, cl_lv = signs_markup(new_sit, self.task.additions[3],
                                                                                               ag_name, size = size, cl_lv=cl_lv)

        agent_state_action = state_prediction(self.task.signs[ag_name], new_sit, self.task.signs)
        conditions_new = get_conditions(new_sit, action, obj.name, ground_block)
        new_sit['conditions'] = conditions_new
        action_situation = define_situation('*goal-sit-*-'+action[1], cell_map_pddl, conditions_new, agent_state_action, self.task.signs)
        action_map = define_map('*goal-map-*-'+action[1], region_map, cell_location, near_loc, self.task.additions[1], self.task.signs)
        state_fixation(action_situation, cell_coords, self.task.signs, 'cell')

        return action_situation, action_map, cl_lv, new_sit, size

    def copy_action(self, act, agent, old_sit):
        from mapspatial.grounding.utils import pm_parser
        new_act = None
        if act[1] == 'Clarify' or act[1] =='Abstract':
            change_name = act[1]
            cell_coords = old_sit.sign.images[2].spread_down_activity_view(1)
            objects = act[-1]['objects']
            map_size = self.problem.map['map-size']
            borders = self.problem.map['wall']
            if change_name == 'Clarify':
                var = cell_coords['cell-4']
                x_size = (var[2] - var[0]) / 6
                y_size = (var[3] - var[1]) / 6
                size = [objects[agent.name]['x'] - x_size, objects[agent.name]['y'] - y_size, objects[agent.name]['x'] + x_size,
                        objects[agent.name]['y'] + y_size]
            else:
                size = [cell_coords['cell-0'][0],
                        cell_coords['cell-0'][1],
                        cell_coords['cell-8'][2],
                        cell_coords['cell-8'][3]]
            rmap = [0, 0]
            rmap.extend(map_size)
            # division into regions and cells
            region_location, region_map = locater('region-', rmap, objects, borders)
            cell_location, cell_map, near_loc, cell_coords, clar_lv = cell_creater(size, objects, region_location,
                                                                                   borders)
            old_sit_new_mean = old_sit.copy('image', 'meaning')

            global SIT_SUF
            SIT_SUF += 1
            conditions, direction, holding = pm_parser(old_sit.sign.images[1], agent.name, self.task.signs, base='image')
            agent_state = state_prediction(agent, direction, holding)
            active_sit_new = define_situation('situation_they_' + str(SIT_SUF), cell_map, conditions, agent_state, self.task.signs)
            active_sit_new_mean = active_sit_new.copy('image', 'meaning')
            new_act = self.task.signs[change_name].add_meaning()
            connector = new_act.add_feature(old_sit_new_mean)
            old_sit.sign.add_out_meaning(connector)
            connector = new_act.add_feature(active_sit_new_mean, effect=True)
            active_sit_new.sign.add_out_meaning(connector)
        else:
            act_mean = act[2].spread_down_activity('meaning', 3)
            act_signs = set()
            for pm_list in act_mean:
                act_signs |= set([c.sign.name for c in pm_list if c.sign.name != 'I'])
            act_signs.add(agent.name)
            action_pms = self.task.signs[act[1]].meanings
            for index, cm in action_pms.items():
                cm_mean = cm.spread_down_activity('meaning', 3)
                cm_signs = set()
                for cm_list in cm_mean:
                    cm_signs |= set([c.sign.name for c in cm_list])
                if act_signs == cm_signs:
                    new_act = cm
                    break
        return new_act

    def load_subtask(self, subtask):
        if subtask[0][0]:
            self.task.actions = ['move', 'rotate', subtask[0][0]]
        else:
            self.task.actions = ['move', 'rotate']
        I_img = self.task.signs['I'].add_image()
        subtask[1].replace('image', self.task.signs[self.name], I_img)
        self.task.goal_situation = subtask[1].sign
        self.task.goal_map = subtask[2].sign
        # Clarification for actions.
        self.task.goal_cl_lv = subtask[-1]
        self.task.initial_state = subtask[3]
        self.task.goal_state = subtask[4]


    def search_solution(self):
        """
        This function is needed to synthesize all plans, choose the best one and
        save the experience.
        """
        logging.info('Поиск плана для проблемы {0} начат в {1}'.format(self.task.name, time.clock()))
        search = SpSearch(self.task, self.task_file, self.backward, self.subsearch)
        solution = search.search_plan()
        # make goal sit be the new start
        self.task.start_situation = self.task.goal_situation
        self.task.start_map = self.task.goal_map
        self.task.init_cl_lv = self.task.goal_cl_lv
        self.task.iteration = max(self.task.additions[0])
        if isinstance(solution[0][-1][-1], dict):
            map_goal = solution[0][-1][-1]
        else:
            map_goal = solution[0][-1][-1][1]
        return solution, map_goal

    def change_start(self, map, act):
        if map != self.task.additions[0][max(self.task.additions[0])]:
            place = self.task.additions[0][max(self.task.additions[0])]['objects'][self.name]
            self.task.additions[0][max(self.task.additions[0])] = map
            self.task.additions[0][max(self.task.additions[0])]['objects'][self.name] = place
            region_map, cell_map_pddl, cell_location, near_loc, cell_coords, size, cl_lv = signs_markup(self.task.additions[0][max(self.task.additions[0])],
                                                                                                        self.task.additions[
                                                                                                            3],
                                                                                                        self.name)
            agent_state_action = state_prediction(self.task.signs[self.name], self.task.additions[0][max(self.task.additions[0])], self.task.signs)
            active_situation = define_situation('*start-sit-*-' + act+'-'+self.name, cell_map_pddl, self.task.additions[0][max(self.task.additions[0])]['conditions'],
                                                agent_state_action, self.task.signs)
            active_map = define_map('*start-map-*-' + act+'-'+self.name, region_map, cell_location, near_loc,
                                    self.task.additions[1], self.task.signs)
            state_fixation(active_situation, cell_coords, self.task.signs, 'cell')
            I_img = self.task.signs['I'].add_image()
            active_situation.replace('image', self.task.signs[self.name], I_img)
            self.task.start_situation = active_situation.sign
            self.task.start_map = active_map.sign

    def get_spatial_finish_blocks(self,cur_sit, cl_lv, size, ag_name):
        sit = deepcopy(self.task.goal_state)
        ag_pose = sit['objects'][self.name]
        sit['objects'] = deepcopy(cur_sit['objects'])
        sit['objects'][self.name] = ag_pose
        region_map, cell_map, cell_location, near_loc, cell_coords, size, cl_lv = signs_markup(sit, self.task.additions[3],
                                                                                               ag_name, size=size, cl_lv=cl_lv)
        agent_state_action = state_prediction(self.task.signs[ag_name], sit, self.task.signs)
        conditions = sit['conditions']
        sit['conditions'] = conditions
        action_situation = define_situation('*goal-sit*-'+ag_name, cell_map, conditions, agent_state_action, self.task.signs)
        action_map = define_map('*goal-map-*-'+ag_name, region_map, cell_location, near_loc, self.task.additions[1], self.task.signs)
        state_fixation(action_situation, cell_coords, self.task.signs, 'cell')

        return action_situation, action_map, cl_lv, sit

class Manager:
    def __init__(self, problem, agpath = 'mapspatial.agent.planning_agent', TaskType = 'spatial', backward = False, subsearch = 'greedy'):
        self.agents = problem.agents
        self.problem = problem
        self.agpath = agpath
        self.agtype = 'SpAgent'
        self.backward = backward
        self.subsearch = subsearch
        self.TaskType = TaskType

    def manage_agents(self):

        allProcesses = []

        for ag in self.agents:
            parent_conn, child_conn = Pipe()
            p = Process(target=agent_activation,
                        args=(self.agpath, self.agtype,ag, self.agents, self.problem, self.backward, self.subsearch, child_conn, ))
            allProcesses.append((p, parent_conn))
            p.start()

        group_experience = []
        for pr, conn in allProcesses:
            group_experience.append((conn.recv(), conn))

        # Select the major (most experienced) agent
        most_exp = 0
        for info, _ in group_experience:
            if info[1] > most_exp:
                most_exp = info[1]

        major = [(info, conn) for info, conn in group_experience if info[1] == most_exp][0]
        others = [(info, conn) for info, conn in group_experience if info[0] != major[0][0]]

        # Major agent will create an auction and send back the best solution.
        for pr, conn in allProcesses:
            conn.send(major[0][0])

        # Solving subtasks
        solution = []
        flag = True
        while flag:
            subtask = major[1].recv()
            if subtask != 'STOP':
                for info, conn in others:
                    if info[0] != subtask[0]:
                        continue
                    conn.send(subtask[1:])
                    solved = conn.recv()
                    major[1].send(solved)
            else:
                solution.extend(major[1].recv())
                for info, conn in others:
                    conn.send('STOP')
                    conn.send(solution)
                flag = False

        for pr, conn in allProcesses:
            pr.join()
        return solution

def agent_activation(agpath, agtype, name, agents, problem, backward, subsearch, childpipe):
    # init agent
    class_ = getattr(importlib.import_module(agpath), agtype)
    workman = class_()
    workman.initialize(name, agents, problem, backward, subsearch)

    # load SWM and calculate the amount of new signs
    task, new_signs = workman.get_task()
    childpipe.send((name, new_signs))

    # load info about the major agent
    major_agent = childpipe.recv()

    # search scenario
    if platform.system() != 'Windows':
        task_paths = problem.task_file.split(delim)[1:-1]
        path = ''.join([delim + el for el in task_paths])
    else:
        task_paths = problem.task_file.split(delim)[:-1]
        path = ''.join([el+delim for el in task_paths[:-1]])
        path += task_paths[-1]
    try:
        pddl_task = path + delim+ 'scenario'+delim+task_paths[-1]+'.pddl'
        open(pddl_task)
    except FileNotFoundError:
        type = problem.name.split(' ')[0]
        pddl_task = os.getcwd() + delim+ 'mapspatial'+delim+'benchmarks'+delim+type+delim\
                    + task_paths[-2] +delim+ task_paths[-1] + delim+ 'scenario' +delim+ task_paths[
            -1] + '.pddl'

    # CALL mapplanner and get pddl solution. But this do only major agent
    flag = True
    solutions = []
    self_solutions = []
    if name == major_agent:
        map = {}
        subtasks = workman.get_scenario(pddl_task, task_paths[-2])
        for sub in subtasks:
            solution = {}
            self_sol = {}
            act_agent = sub[0][-1]
            if act_agent == name or act_agent == 'I':
                if map:
                    workman.change_start(map, sub[0][1])
                workman.load_subtask(sub)
                subtask_solution, map = workman.search_solution()
                if isinstance(subtask_solution[0], list):
                    subtask_solution = subtask_solution[0]
                minor_message = []
                for action in subtask_solution:
                    minor_message.append((None, action[1], None, None, (None, None), (None, None), deepcopy(action[6])))
                solution[sub[0]] = subtask_solution
                self_sol[sub[0]] = minor_message
                solutions.append(self_sol)
                self_solutions.append((self_sol, solution))
            else:
                childpipe.send((act_agent, sub, map))
                ag_solution = childpipe.recv()
                solution[sub[0]] = ag_solution[0]
                solutions.append(solution)
                map = ag_solution[1]
        childpipe.send('STOP')
        childpipe.send(solutions)
    else:
        while flag:
            subtask = childpipe.recv()
            if subtask == 'STOP':
                major_solutions = childpipe.recv()
                if major_solutions:
                    major_agent_sign = workman.task.signs[major_agent]
                    for subplan in major_solutions:
                        solution = {}
                        for act_descr, ag_solution in subplan.items():
                            if act_descr[1] == 'I':
                                act_descr_new = (act_descr[0], major_agent_sign.name)
                            elif act_descr[1] == name:
                                act_descr_new = (act_descr[0], 'I')
                            else:
                                act_descr_new = act_descr
                            solution[act_descr_new] = ag_solution
                            solutions.append(solution)
                    logging.info("Конечное решение получено агентом {0}".format(name))
                else:
                    logging.debug('Agent {0} cant load the major solution'.format(name))
                flag = False
            else:
                if subtask[1]:
                    workman.change_start(subtask[1], subtask[0][0][1])
                workman.load_subtask(subtask[0])
                subtask_solution, map = workman.search_solution()
                if isinstance(subtask_solution[0], list):
                    subtask_solution = subtask_solution[0]
                solution = {}
                pddl_name = (subtask[0][0][0], 'I')
                solution[pddl_name] = subtask_solution
                self_sol = {}
                major_message = []
                for action in subtask_solution:
                    major_message.append((None, action[1], None, None, (None, None), (None, None), deepcopy(action[6])))
                self_sol[subtask[0][0]] = major_message
                self_solutions.append((self_sol, solution))
                if subtask_solution:
                    childpipe.send((major_message, map))
                else:
                    logging.info("Агент {0} не смог синтезировать план".format(name))

    for ind, act1 in enumerate(copy(solutions)):
        for act1_name, act1_map in act1.items():
            for act2, self_act in self_solutions:
                flag = False
                for act2_name, act2_map in act2.items():
                    if act1_name[0] == act2_name[0]:
                        if act1_map == act2_map:
                            solutions[ind] = self_act
                            flag = True
                            break
                if flag:
                    break
    file_name = workman.task.save_signs(solutions)

    if file_name:
        logging.info('Агент ' + name + ' закончил работу')


def get_conditions(new_sit, action, obj, ground_block):
    conditions_new = {}
    if action[1] == 'pick-up':
        for pred, signature in new_sit['conditions'].items():
            if obj not in signature['cause']:
                conditions_new[pred] = signature
            elif 'blocktype' in pred:
                conditions_new[pred] = signature
    elif action[1] == 'put-down':
        pred_num = 0
        for pred, signature in new_sit['conditions'].items():
            if obj not in signature['cause']:
                conditions_new[pred] = signature
            elif 'blocktype' in pred:
                conditions_new[pred] = signature
                pred_num = pred.split('-')[-1]
        conditions_new['clear-'+pred_num] = {'cause':[obj], 'effect':[]}
        conditions_new['onground-' + pred_num] = {'cause': [obj], 'effect': []}
    elif action[1] == 'stack':
        on_num = 0
        for pred, signature in new_sit['conditions'].items():
            if obj not in signature['cause'] and ground_block.name not in signature['cause']:
                conditions_new[pred] = signature
            elif 'blocktype' in pred:
                conditions_new[pred] = signature
                if obj in signature['cause']:
                    on_num = pred.split('-')[-1]
            elif ground_block.name in signature['cause'] and 'clear' not in pred:
                conditions_new[pred] = signature
        conditions_new['clear-'+on_num] = {'cause':[obj], 'effect':[]}
        conditions_new['on-' + on_num] = {'cause': [obj, ground_block.name], 'effect': []}
    elif action[1] == 'unstack':
        und_num = 0
        for pred, signature in new_sit['conditions'].items():
            if obj not in signature['cause']:
                conditions_new[pred] = signature
                if ground_block.name in signature['cause']:
                    und_num = pred.split('-')[-1]
            elif 'blocktype' in pred:
                conditions_new[pred] = signature
        conditions_new['clear-'+und_num] = {'cause':[obj], 'effect':[]}
    return conditions_new