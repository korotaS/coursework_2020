import math
import subprocess
import json

from mapcore.planning.search.mapsearch import *
from mapspatial.grounding import planning_task as st
from mapspatial.grounding.utils import *


MAX_CL_LV = 1

class SpSearch(MapSearch):
    def __init__ (self, task, task_file, backward, subsearch, init_state=None, goal_state=None):
        super().__init__(task,'spatial', backward)
        self.MAX_ITERATION = 30
        if self.backward:
            self.goal_pm = task.start_situation.images[1]
            self.active_pm = task.goal_situation.images[1]
            self.goal_map = task.start_map.images[1]
            self.active_map = task.goal_map.images[1]
        else:
            self.goal_pm = task.goal_situation.images[1]
            self.active_pm = task.start_situation.images[1]
            self.active_map = task.start_map.images[1]
            self.goal_map = task.goal_map.images[1]
        self.check_map = False

        logging.debug('Start: {0}'.format(self.goal_pm.longstr()))
        logging.debug('Finish: {0}'.format(self.active_pm.longstr()))

        self.additions = task.additions
        # self.agents = set()
        self.I_sign = self.world_model['I']
        I_objects = [con.in_sign for con in self.I_sign.out_significances if con.out_sign.name == "I"]
        if I_objects:
            self.I_obj = I_objects[0]
        else:
            self.I_obj = None
        if init_state is None:
            self.init_state = task.initial_state
        else:
            self.init_state = init_state
        if goal_state is None:
            self.goal_state = task.goal_state
        else:
            self.goal_state = goal_state
        if task.init_cl_lv is None:
            self.clarification_lv = self.additions[0][0][self.I_obj.name]['cl_lv_init']
        else:
            self.clarification_lv = task.init_cl_lv
        # Changed while scenario action was created
        self.goal_cl_lv = task.goal_cl_lv
        self.exp_sits = []
        self.exp_maps = []
        self.exp_acts = {}
        self.task_file = task_file
        self.subsearch = subsearch
        self.precedents = set()
        self.subtasks = task.subtasks
        self.actions = task.actions
        self.iteration = task.iteration


    def search_plan(self):
        self._precedent_activation()
        plans = self._map_sp_iteration(self.active_pm, self.active_map, iteration=self.iteration, current_plan=[])
        if self.backward:
            plans = [list(reversed(plan[0])) for plan in plans]
        return plans

    def _map_sp_iteration(self, active_pm, active_map, iteration, current_plan, prev_state = [], goal_pm = None, goal_map = None):
        logging.debug('STEP {0}:'.format(iteration))
        logging.debug('\tSituation {0}'.format(active_pm.longstr()))

        MAX_ITERATION = self.MAX_ITERATION

        if iteration >= MAX_ITERATION:
            logging.debug('\tMax iteration count')
            return None

        precedents = self._precedent_search(active_pm)

        active_pm, active_map, goal_pm, goal_map = self.state_activation(active_pm, active_map, goal_pm, goal_map)

        active_chains = active_pm.spread_down_activity('meaning', 4)
        active_signif = set()

        for chain in active_chains:
            pm = chain[-1]
            active_signif |= pm.sign.spread_up_activity_act('significance', 3)

        if self.actions:
            active_signif = {act for act in active_signif if act.sign.name in self.actions}

        meanings = []
        for pm_signif in active_signif:
            chains = pm_signif.spread_down_activity('significance', 6)
            merged_chains = []
            for chain in chains:
                for achain in active_chains:
                    if chain[-1].sign == achain[-1].sign and len(chain) > 2 and chain not in merged_chains:
                        merged_chains.append(chain)
                        break
            scripts = self._generate_meanings(merged_chains)
            meanings.extend(scripts)

        applicable_meanings = self.applicable_sp_search(precedents + meanings, active_pm, iteration)

        prev_act = None
        if current_plan:
            prev_act = current_plan[-1][1]

        candidates = self._sp_check_activity(active_pm, applicable_meanings, [x for x, _, _, _, _,_,_ in current_plan], iteration, prev_state, prev_act)

        if not candidates:
            logging.debug('\tNot found applicable scripts ({0})'.format([x for _, x, _, _, _,_,_ in current_plan]))
            return None

        logging.debug('\tFound {0} variants'.format(len(candidates)))
        final_plans = []

        if candidates[0][0] == 0:
            # there are no actions that let to achieve the goal
            current_plans, active_pm, active_map, iteration = self.clarify_search(self.I_obj.name, active_pm, goal_pm, iteration, current_plan)
            candidates = []
            final_plans.extend(current_plans)
            current_plan = current_plans[0]
        elif self.goal_cl_lv > self.clarification_lv and \
                [cand for cand in candidates if cand[2].sign.name != 'move' and cand[2].sign.name != 'rotate' and 'subplan' not in cand[2].sign.name]:
            # there task is more
            current_plan, active_pm, active_map, iteration = self.clarify_search(self.I_obj.name, active_pm, goal_pm, iteration, current_plan)
            return current_plan

        logging.info("Текущая длина найденного плана: {0}. Количество возможных действий: {1}".format(len(current_plan), len(candidates)))

        for counter, name, script, ag_mask, _ in candidates:
            logging.debug('\tChoose {0}: {1} -> {2}'.format(counter, name, script))
            plan = copy(current_plan)

            subplan = None

            next_pm, next_map, prev_state, direction = self._step_generating(active_pm, active_map, script, self.I_sign, iteration, prev_state, True)

            if name != 'move':
                ag_place = self.additions[0][iteration]['objects'][self.I_obj.name]
            else:
                ag_place = self.additions[0][iteration+1]['objects'][self.I_obj.name]
            if script in [pr[1] for pr in precedents]:
                acts = []
                for act in script.sign.images[1].spread_down_activity('image', 2):
                    if act[1] not in acts:
                        acts.append(act[1])
                self.exp_sits.append(next_pm)
                self.exp_maps.append(active_map.sign.images[1])
                self.exp_maps.append(self.goal_map.sign.images[1])
                if self.backward:
                    self.backward = False
                    maxkey = max(self.additions[0].keys())
                    self.additions[0][maxkey], self.additions[0][maxkey-1] = self.additions[0][maxkey-1], self.additions[0][maxkey]
                    self.additions[2]["I"][maxkey], self.additions[2]["I"][maxkey - 1] = self.additions[2]["I"][maxkey - 1], \
                                                                               self.additions[2]["I"][maxkey]
                    subplan = self.hierarchical_exp_sp_search(next_pm, next_map, active_pm.sign.images[1], active_map.sign.images[1], iteration, prev_state, acts)
                    subplan = list(reversed(subplan))
                    values = list(reversed(list(self.additions[0].values())))
                    for num, _ in copy(self.additions[0]).items():
                        self.additions[0][num] = values[num]
                    values2 = list(reversed(list(self.additions[0].values())))
                    for num, _ in copy(self.additions[2]["I"]).items():
                        self.additions[2]["I"][num] = values2[num]
                    self.backward = True
                else:
                    subplan = self.hierarchical_exp_sp_search(active_pm.sign.images[1], active_map.sign.images[1],
                                                                  next_pm, next_map, iteration, prev_state, acts)

            if not subplan:
                plan.append((active_pm.sign.images[1], name, script, ag_mask, (ag_place, direction),
                             (active_map, self.clarification_lv), (self.additions[0][iteration], self.additions[0][iteration+1])))
            else:
                plan.extend(subplan)
                logging.info(
                    'Сложное действие {0} уточнено. Найденные поддействия: {1}'.format(script.sign.name, [part[1] for part in subplan]))
            if self.clarification_lv > 0:
                # there are some actions that let to achieve the goal, check the higher lev of hierarchy
                next_pm, goal_pm, next_map, goal_map, iteration, plan = self.abstract_search(self.I_obj.name, next_pm, goal_pm, next_map, goal_map, iteration, plan)


            if next_pm.includes('image', goal_pm.sign.images[1]):
                if self.check_map:
                    if next_map.sign.images[1].includes('image', goal_map.sign.images[1]):
                        flag = True
                    else:
                        flag = False
                else:
                    flag = True
                if flag:
                    final_plans.append(plan)
                    plan_actions = [x.sign.name for _, _, x, _, _, _, _ in plan]
                    logging.info("Цель достигнута. Длина найденного плана: {0}".format(len(plan)))
                    logging.info(plan_actions)
                else:
                    recursive_plans = self._map_sp_iteration(next_pm, next_map, iteration + 1, plan, prev_state,
                                                          goal_pm=goal_pm,
                                                          goal_map=goal_map)
                    if recursive_plans:
                        final_plans.extend(recursive_plans)
            else:
                recursive_plans = self._map_sp_iteration(next_pm, next_map, iteration + 1, plan, prev_state, goal_pm= goal_pm, goal_map=goal_map)
                if recursive_plans:
                    final_plans.extend(recursive_plans)


        return final_plans

    def _precedent_search(self, active_pm):
        precedents = []
        active_cm = active_pm.copy('image', 'meaning')
        for cm in self.precedents:
            # Todo: use plan_ signs to check experienced agents
            if cm.sign.name.startswith('plan_'):
                continue
            result, checked = self._check_activity(cm, active_cm, self.backward, True)
            if result:
                precedents.append((self.I_sign, checked))
        return precedents

    def _precedent_activation(self):
        if not self.exp_sits:
            self.exp_sits = list(self.world_model['situation'].spread_down_activity_obj('image', 1))
        if not self.exp_maps:
            self.exp_maps = [sign.images[1] for name, sign in self.world_model.items() if
                             'exp_' in name and 'map' in name]
        if not self.exp_acts:
            self.exp_acts = self.hierarch_acts()

        for sit in self.exp_sits:
            if sit.sign.out_meanings:
                precedent = sit.sign.spread_up_activity_act('meaning', 1)
                if precedent:
                    pr = list(precedent)[0].sign
                    if not pr.name == 'Abstract' and not pr.name == 'Clarify':
                        self.precedents.add(pr.meanings[1])

    def applicable_sp_search(self, meanings, active_pm, iteration):
        applicable_meanings = set()
        for agent, cm in meanings:
            expandable = True
            if cm.sign.name in self.actions:
                expandable = False
            if not self.backward:
                result, checked = self._check_activity(cm, active_pm.sign.meanings[1], expandable=expandable)
            else:
                result, checked = self._check_activity_backward_spat(cm, active_pm.sign.meanings[1], agent, iteration)
            if result:
                applicable_meanings.add((agent, checked))
        return applicable_meanings

    def state_activation(self, act_pm, act_map, ch_pm, ch_map):
        if not ch_pm and not ch_map:
            check_pm= self.goal_pm
            check_map = self.goal_map
        else:
            check_pm = ch_pm
            check_map = ch_map

        if not check_pm.sign.meanings:
            check_pm_mean = check_pm.copy('image', 'meaning')
        else:
            check_pm_mean = check_pm.sign.meanings[1]
        if not check_map.sign.meanings:
            check_map_mean = check_map.sign.images[1].copy('image', 'meaning')
        else:
            check_map_mean = check_map.sign.meanings[1]
        if not act_pm.sign.meanings:
            act_pm_mean = act_pm.copy('image', 'meaning')
        else:
            act_pm_mean = act_pm.sign.meanings[1]
        if not act_map.sign.meanings:
            act_map_mean = act_map.sign.images[1].copy('image', 'meaning')
        else:
            act_map_mean = act_map.sign.meanings[1]

        return act_pm_mean, act_map_mean, check_pm_mean, check_map_mean

    def hierarchical_exp_sp_search(self, active_pm, active_map, goal_pm, goal_map, iteration, prev_state, acts, cur_plan = [], subsearch = False, cycle = 0):
        """
        create a subplan using images info
        :param script: parametrs to generate plan
        :return:plan
        """
        logging.info('Clarify experience plan')
        active_pm, active_map, goal_pm, goal_map = self.state_activation(active_pm, active_map, goal_pm, goal_map)
        applicable = []
        act = None
        while len(cur_plan) < len(acts):
            act = acts[cycle].sign
            break

        finall_plans = []

        if act:
            exp_acts = [act[1] for act in self.exp_acts[act]]
            for exp_act in exp_acts:
                result = False
                checked = exp_act
                try:
                    if exp_act.sign.name.startswith('subplan_') or exp_act.sign.name.startswith('Clarify')  \
                            or exp_act.sign.name.startswith('Abstract') :
                        result, checked = self._check_activity(exp_act, active_pm)
                    else:
                        result, checked = self._check_activity(exp_act, active_pm, expandable=False)
                except Exception:
                    # delete expanded simple actions
                    self.exp_acts = [act for act in self.exp_acts[act] if act[1] != exp_act]
                if result:
                    applicable.append((self.world_model["I"], checked))
        else:
            finall_plans = cur_plan

        for action in applicable:
            plan = copy(cur_plan)
            if action[1].sign.name == 'Clarify':
                result = False
                next_pm = False
                next_map = None
                while not result:
                    if self.clarification_lv <= MAX_CL_LV:
                        next_pm, check_pm, next_map, check_map, _ = self.devide_situation(active_pm, goal_pm,
                                                                                                  iteration, self.I_obj.name)
                    else:
                        break
                    if not next_pm.sign.meanings:
                        next_mm = next_pm.copy("image", "meaning")
                    else:
                        next_mm = next_pm.sign.meanings[1]
                    result, checked = self._check_result(action[1], next_mm)

                cell = []
                side = None
                if cur_plan:
                    agent = plan[-1][3]
                    place = plan[-1][4]
                else:
                    agent = action[0]
                    orient = next_pm.get_iner(self.world_model['orientation'], 'image')[0]

                    for sign in orient.get_signs():
                        if sign.name != agent.name and sign.name != 'I':
                            side = sign.name
                    cell = next_pm.sign.images[2].spread_down_activity_view(1)['cell-4']
                    ag_coords = cell[0] + ((cell[2] - cell[0]) // 2), cell[1] + ((cell[3] - cell[1]) // 2)
                    place = (ag_coords, side)
                if max(self.additions[0]) > 0:
                    descr = max(self.additions[0]) - 1
                else:
                    descr = max(self.additions[0])
                plan.append((active_pm.sign.images[1], action[1].sign.name, action[1], agent, place, (next_map.sign.images[1], self.clarification_lv), self.additions[0][descr]))
                print("{0}. act: Clarify, cell: {1}, dir: {2}".format(str(cycle), cell,
                                                                            side))

            elif action[1].sign.name == 'Abstract':
                result = False
                next_pm = False
                next_map = None
                while not result:
                    next_pm, _, next_map, _, _, _ = self.combine_situation(active_pm, active_map, goal_pm, goal_map, iteration, self.I_obj.name)
                    if not next_pm.sign.meanings:
                        next_mm = next_pm.copy("image", "meaning")
                    else:
                        next_mm = next_pm.sign.meanings[1]
                    result, checked = self._check_result(action[1], next_mm)

                cell = []
                side = None
                if cur_plan:
                    agent = plan[-1][3]
                    place = plan[-1][4]
                else:
                    agent = action[0]
                    orient = next_pm.get_iner(self.world_model['orientation'], 'image')[0]

                    for sign in orient.get_signs():
                        if sign.name != agent.name and sign.name != 'I':
                            side = sign.name
                    cell = next_pm.sign.images[2].spread_down_activity_view(1)['cell-4']
                    ag_coords = cell[0] + ((cell[2] - cell[0]) // 2), cell[1] + ((cell[3] - cell[1]) // 2)
                    place = (ag_coords, side)
                if max(self.additions[0]) > 0:
                    descr = max(self.additions[0]) - 1
                else:
                    descr = max(self.additions[0])
                plan.append((active_pm.sign.images[1], action[1].sign.name, action[1], agent, place, (next_map.sign.images[1], self.clarification_lv), self.additions[0][descr]))
                print("{0}. act: Abstract, cell: {1}, dir: {2}".format(str(cycle), cell,
                                                                            side))

            elif 'subplan_' in action[1].sign.name:
                sub_sign = action[1].sign
                sub_acts = []
                for act in sub_sign.images[1].spread_down_activity('image', 2):
                    if act[1] not in sub_acts:
                        sub_acts.append(act[1])
                sub_finish = None
                sub_start = None

                if len(acts) > 1:
                    for con in acts[1].sign.meanings[1].cause[0].coincidences:
                        sub_finish = con.out_sign.meanings[con.out_index]
                else:
                    for con in sub_sign.meanings[1].effect[0].coincidences:
                        sub_finish = con.out_sign.meanings[con.out_index]


                for con in sub_sign.meanings[1].cause[0].coincidences:
                    sub_start = con.out_sign.meanings[con.out_index]


                plan = self.hierarchical_exp_sp_search(sub_start.sign.images[1], active_map.sign.images[1], sub_finish.sign.images[1], None, iteration,
                                                        prev_state, sub_acts, plan, True, cycle=cycle+1)
                next_pm = sub_finish
                next_map = plan[-1][-1][0]
                if plan:
                    subsearch = False

            else:
                if goal_map or subsearch:
                    next_pm, next_map, prev_state, direction = self._step_generating(active_pm, active_map, action[1], action[0],
                                                                                     iteration, prev_state, True)
                    ag_place = (prev_state[-1][2] - prev_state[-1][0]) // 2 + prev_state[-1][0], (
                            prev_state[-1][3] - prev_state[-1][1]) // 2 + prev_state[-1][1]


                    included_sit = [sit for sit in self.exp_sits if sit.includes('image', next_pm)]

                    if included_sit:
                        if plan is None: plan = []
                        plan.append((active_pm.sign.images[1], action[1].sign.name, action[1], action[0], (ag_place, direction),
                                     (next_map, self.clarification_lv), (self.additions[0][max(self.additions[0])-1], self.additions[0][max(self.additions[0])])))
                    else:
                        continue
                else:
                    next_pm = self._time_shift_spat(active_pm, action[1])
                    included_map = True
                    next_map = None
                    included_sit = [sit for sit in self.exp_sits if sit.includes('image', next_pm)]
                    if included_sit and included_map:
                        plan.append(
                            (active_pm, action[1].sign.name, action[1], action[0], None, (None, self.clarification_lv), (self.additions[0][max(self.additions[0])-1], self.additions[0][max(self.additions[0])])))


            if next_pm.sign.images[1].includes('image', goal_pm.sign.images[1]):
                if goal_map:
                    if next_map.sign.images[1].includes('image', goal_map.sign.images[1]):
                        finall_plans.extend(plan)
                        break
                    else:
                        plan = self.hierarchical_exp_sp_search(next_pm.sign.images[1], next_map.sign.images[1], goal_pm.sign.images[1], goal_map.sign.images[1], iteration + 1,
                                                            prev_state, acts, plan, False, cycle=cycle+1)
                        if plan:
                            finall_plans.extend(plan)
                            break
                elif subsearch:
                    if len(acts):
                        plan =  self.hierarchical_exp_sp_search(next_pm.sign.images[1], next_map.sign.images[1], goal_pm.sign.images[1], goal_map.sign.images[1], iteration + 1,
                                                            prev_state, acts, plan, subsearch, cycle=cycle+1)
                        if plan:
                            finall_plans.extend(plan)
                    else:
                        finall_plans.append(plan)
                        break
            else:
                plan = self.hierarchical_exp_sp_search(next_pm.sign.images[1], next_map.sign.images[1], goal_pm.sign.images[1], goal_map.sign.images[1],
                                                       iteration+1, prev_state, acts, plan, subsearch, cycle=cycle+1)
                if plan:
                    finall_plans.extend(plan)
                    break
        return finall_plans

    def combine_situation(self, active_pm, active_map, goal_pm, goal_map, iteration, agent):
        # define new start situation
        objects = self.additions[0][iteration]['objects']
        map_size = self.additions[3]['map-size']
        borders = self.additions[3]['wall']
        orientation = self.additions[0][iteration][self.I_obj.name]['orientation']
        cell_coords = active_pm.sign.images[2].spread_down_activity_view(1)
        rmap = [0, 0]
        rmap.extend(map_size)
        region_location, region_map = locater('region-', rmap, objects, borders)
        size = [cell_coords['cell-0'][0],
                cell_coords['cell-0'][1],
                cell_coords['cell-8'][2],
                cell_coords['cell-8'][3]]
        x_size = size[2] - size[0]
        y_size = size[3] - size[1]
        if x_size > map_size[0] // 3 or y_size > map_size[1] // 3:
            agplx = self.additions[0][iteration]['objects'][agent]['x']
            agply = self.additions[0][iteration]['objects'][agent]['y']
            for _, rsize in region_location.items():
                if rsize[0] <= agplx <= rsize[2] and rsize[1] <= agply <= rsize[3]:
                    size = rsize
                    break

        # combining into regions and cells
        cell_location, cell_map, near_loc, cell_coords, clar_lv = cell_creater(size, objects, region_location, borders)

        # check the stright forward path
        front_cell = None
        for reg, val in self.additions[1]['region-4'].items():
            if val[1] == orientation:
                front_cell = 'cell-' + reg[-1]
                break
        #path to goal
        coords = objects[self.I_obj.name]['x'],objects[self.I_obj.name]['y']
        if self.backward:
            g = self.init_state
        else:
            g = self.goal_state
        gcoords = g['objects'][self.I_obj.name]['x'], g['objects'][self.I_obj.name]['y']
        gpath = math.sqrt((gcoords[1]-coords[1])**2 + (gcoords[0]-coords[0])**2)
        if not 0 in cell_map[front_cell] and not self.backward:
            if 'wall' not in list(cell_map[front_cell])[0]:
                logging.debug('Aбстрагирование запрещено. Путь до цели содержит препятствия.')
                return active_pm.sign.images[1], goal_pm.sign.images[1], active_map.sign.images[1], goal_map.sign.images[1], iteration, False
        elif x_size //3//2 <= gpath <= x_size//3 or y_size //3//2 <= gpath <= y_size//3:
            logging.debug("Абстрагирование запрещено. Целевое состояние слишком близко.")
            return active_pm.sign.images[1], goal_pm.sign.images[1], active_map.sign.images[1], goal_map.sign.images[
                1], iteration, False
        # say that we are on the prev lv of hierarchy
        self.clarification_lv -= 1
        logging.info('Абстрагирование разрешено. Новый уровень уточнения ситуации: {0}'.format(self.clarification_lv))

        sit_name = st.SIT_PREFIX + str(st.SIT_COUNTER)
        st.SIT_COUNTER += 1
        conditions, direction, holding = pm_parser(active_pm.sign.images[1], agent, self.world_model, base='image')
        agent_state = state_prediction(self.world_model['I'], direction, self.world_model, holding)
        active_sit_new = define_situation(sit_name + 'sp', cell_map, conditions, agent_state, self.world_model)
        active_map_new = define_map(st.MAP_PREFIX + str(st.SIT_COUNTER), region_map, cell_location, near_loc,
                                self.additions[1],
                                self.world_model)
        st.SIT_COUNTER += 1
        iteration += 1
        state_fixation(active_sit_new, cell_coords, self.world_model, 'cell')
        self.additions[0][iteration] = deepcopy(self.additions[0][iteration - 1])
        self.additions[0][iteration]['cl_lv'] = self.clarification_lv
        self.additions[2]["I"][iteration] = cell_map

        if self.backward:
            goal = self.init_state
        else:
            goal = self.goal_state

        # define new finish situation or check finish achievement
        if self.clarification_lv == self.goal_cl_lv:
            logging.info('Достигнут уровень уточнения цели с помощью абстрагирования. Уровень уточнения : {0}'.format(self.clarification_lv))
            goal_sit_new = self.goal_pm.sign.images[1]
        else:
            sit_name = st.SIT_PREFIX + str(st.SIT_COUNTER)
            region_location, region_map = locater('region-', rmap, goal['objects'], borders)
            cell_location, cell_map, near_loc, cell_coords, clar_lv = cell_creater(size,
                                                                                   goal['objects'],
                                                                                   region_location, borders)

            conditions, direction, holding = pm_parser(goal_pm.sign.images[1], agent, self.world_model,base='image')
            agent_state = state_prediction(self.world_model['I'], direction, self.world_model, holding)
            goal_sit_new = define_situation(sit_name + 'sp', cell_map, conditions, agent_state, self.world_model)
            goal_map = define_map(st.MAP_PREFIX + str(st.SIT_COUNTER), region_map, cell_location, near_loc,
                                  self.additions[1],
                                  self.world_model)
            st.SIT_COUNTER += 1
            state_fixation(goal_sit_new, cell_coords, self.world_model, 'cell')

        return active_sit_new, goal_sit_new, active_map_new, goal_map, iteration, True

    def abstract_search(self, agent, active_pm, goal_pm, active_map, goal_map, iteration, plan):
        logging.debug('Начата проверка возможности абстрагирования карты...')
        # checking for empty situation
        first_check = True
        iteration+=1
        for cell, value in self.additions[2]["I"][iteration].items():
            if not 0 in value:
                if not agent in value:
                    first_check = False
                    logging.debug('Абстрагирование не разрешено. В фокусе внимания присутствуют объекты.')
                    break
        if first_check:
            # calculate new start situation
            if self.clarification_lv <= self.goal_cl_lv:
                logging.debug('Абстрагирование запрещено. Достигнут уровень уточнения цели.')
                return active_pm, goal_pm, active_map, goal_map, iteration-1, plan
            active_sit_new, goal_sit_new, active_map_new, goal_map_new, iteration, reply = \
                self.combine_situation(active_pm, active_map, goal_pm, goal_map, iteration, agent)

            if reply is False:
                return active_pm, goal_pm, active_map, goal_map, iteration - 1, plan

            act_descr = deepcopy(self.additions[0][iteration])
            # because it is descr of pre-clarified situation
            act_descr['cl_lv'] = self.clarification_lv + 1

            if self.backward:
                # devide sit while planning backward
                change_name = 'Clarify'
                if not active_pm.sign.meanings:
                    active_mean = active_pm.copy('image', 'meaning')
                else:
                    active_mean = active_pm.sign.meanings[1]
                active_sit_new_mean = active_sit_new.copy('image', 'meaning')
                change_mean = self.world_model[change_name].add_meaning()
                connector = change_mean.add_feature(active_mean, effect=True)
                active_mean.sign.add_out_meaning(connector)
                connector = change_mean.add_feature(active_sit_new_mean)
                active_sit_new.sign.add_out_meaning(connector)
                sit_to_plan = active_sit_new.sign.images[1]
            else:
                # create Abstract action while planning forward
                change_name = 'Abstract'
                if not active_pm.sign.meanings:
                    active_mean = active_pm.copy('image', 'meaning')
                else:
                    active_mean = active_pm.sign.meanings[1]
                active_sit_new_mean = active_sit_new.copy('image', 'meaning')
                change_mean = self.world_model[change_name].add_meaning()
                connector = change_mean.add_feature(active_mean)
                active_mean.sign.add_out_meaning(connector)
                connector = change_mean.add_feature(active_sit_new_mean, effect=True)
                active_sit_new.sign.add_out_meaning(connector)
                sit_to_plan = active_pm.sign.images[1]

            plan.append(
                (sit_to_plan, change_name, change_mean, self.world_model['I'], plan[-1][4], (plan[-1][5][0], plan[-1][5][1]-1), act_descr))

            return active_sit_new, goal_sit_new, active_map_new, goal_map_new, iteration-1, plan

        return active_pm, goal_pm, active_map, goal_map, iteration-1, plan

    def abstract(self, active_pm, agent, iteration):
        objects = self.additions[0][iteration]['objects']
        map_size = self.additions[3]['map-size']
        borders = self.additions[3]['wall']
        #orientation = self.additions[0][iteration][self.I_obj.name]['orientation']
        rmap = [0, 0]
        rmap.extend(map_size)
        region_location, region_map = locater('region-', rmap, objects, borders)
        cell_coords = active_pm.sign.images[2].spread_down_activity_view(1)
        size = [cell_coords['cell-0'][0],
                cell_coords['cell-0'][1],
                cell_coords['cell-8'][2],
                cell_coords['cell-8'][3]]
        x_size = size[2] - size[0]
        y_size = size[3] - size[1]
        if x_size > map_size[0] // 3 or y_size > map_size[1] // 3:
            agplx = self.additions[0][iteration]['objects'][agent]['x']
            agply = self.additions[0][iteration]['objects'][agent]['y']
            for _, rsize in region_location.items():
                if rsize[0] <= agplx <= rsize[2] and rsize[1] <= agply <= rsize[3]:
                    size = rsize
                    break
        cell_location, cell_map, near_loc, cell_coords, _ = cell_creater(size, objects, region_location, borders)
        # define new sit
        sit_name = st.SIT_PREFIX + str(st.SIT_COUNTER)
        events, direction, holding = pm_parser(active_pm, agent, self.world_model)
        agent_state = state_prediction(self.world_model['I'], direction, self.world_model, holding)
        goal_pm = define_situation(sit_name + 'sp', cell_map, events, agent_state, self.world_model)
        # define new map
        goal_map = define_map(st.MAP_PREFIX + str(st.SIT_COUNTER), region_map, cell_location, near_loc,
                                self.additions[1],
                                self.world_model)
        st.SIT_COUNTER += 1

        state_fixation(goal_pm, cell_coords, self.world_model, 'cell')

        # decrease clarification
        self.clarification_lv -= 1

        return goal_pm, goal_map, cell_map


    def devide_situation(self, active_pm, check_pm, iteration, agent):
        # say that we are on the next lv of hierarchy
        self.clarification_lv+=1
        logging.info('Уточнение ситуации. Уровень уточнения: {0}'.format(self.clarification_lv))

        #define new start situation
        var = active_pm.sign.images[2].spread_down_activity_view(1)['cell-4']
        objects = self.additions[0][iteration]['objects']
        map_size = self.additions[3]['map-size']
        borders = self.additions[3]['wall']
        x_size = (var[2] - var[0]) / 6
        y_size = (var[3] - var[1]) / 6
        size = [objects[agent]['x'] - x_size, objects[agent]['y'] - y_size, objects[agent]['x'] + x_size, objects[agent]['y'] + y_size]
        rmap = [0, 0]
        rmap.extend(map_size)
        # division into regions and cells
        region_location, region_map = locater('region-', rmap, objects, borders)
        cell_location, cell_map, near_loc, cell_coords, clar_lv = cell_creater(size, objects, region_location, borders)
        self.clarification_lv += clar_lv

        sit_name = st.SIT_PREFIX + str(st.SIT_COUNTER)
        st.SIT_COUNTER += 1
        conditions, direction, holding = pm_parser(active_pm.sign.images[1], agent, self.world_model, base='image')
        agent_state = state_prediction(self.world_model['I'], direction, self.world_model, holding)
        active_sit_new = define_situation(sit_name + 'sp', cell_map, conditions, agent_state, self.world_model)
        active_map = define_map(st.MAP_PREFIX + str(st.SIT_COUNTER), region_map, cell_location, near_loc,
                                self.additions[1],
                                self.world_model)
        st.SIT_COUNTER += 1
        iteration+=1
        state_fixation(active_sit_new, cell_coords, self.world_model, 'cell')
        self.additions[0][iteration] = deepcopy(self.additions[0][iteration-1])
        self.additions[2]["I"][iteration] = cell_map

        if self.backward:
            goal = self.init_state
        else:
            goal = self.goal_state

        if self.clarification_lv == self.goal_cl_lv:
            logging.debug('Достигнут уровень уточнения, как в целевой ситуации: {0}'.format(self.clarification_lv))
        if self.clarification_lv > self.goal_cl_lv:
            sit_name = st.SIT_PREFIX + str(st.SIT_COUNTER)
            goal_size = [goal['objects'][agent]['x'] - x_size, goal['objects'][agent]['y'] - y_size, goal['objects'][agent]['x'] + x_size, goal['objects'][agent]['y'] + y_size]
            region_location, region_map = locater('region-', rmap, goal['objects'], borders)
            cell_location, cell_map, near_loc, cell_coords, clar_lv = cell_creater(goal_size, goal['objects'], region_location, borders)
            events, direction, holding = pm_parser(check_pm.sign.images[1], agent, self.world_model)
            agent_state = state_prediction(self.world_model['I'], direction, self.world_model, holding)
            goal_sit_new = define_situation(sit_name + 'sp', cell_map, events, agent_state, self.world_model)
            goal_map = define_map(st.MAP_PREFIX + str(st.SIT_COUNTER), region_map, cell_location, near_loc,
                                    self.additions[1],
                                    self.world_model)
            st.SIT_COUNTER += 1
            state_fixation(goal_sit_new, cell_coords, self.world_model, 'cell')
        else:
            goal_sit_new = self.goal_pm
            goal_map = self.goal_map

        return active_sit_new, goal_sit_new, active_map, goal_map, iteration

    def clarify_search(self, agent, active_pm, check_pm, iteration, current_plan):

        active_sit_new, goal_sit_new, active_map, goal_map, iteration = self.devide_situation(active_pm, check_pm,
                                                                                              iteration, agent)
        act_descr = deepcopy(self.additions[0][iteration])
        #because it is descr of pre-clarified situation
        act_descr['cl_lv'] = self.clarification_lv - 1

        if not self.backward:
            #devide sit while planning forward
            change_name = 'Clarify'
            active_sit_new_mean = active_sit_new.copy('image', 'meaning')
            change_mean = self.world_model[change_name].add_meaning()
            connector = change_mean.add_feature(active_pm)
            active_pm.sign.add_out_meaning(connector)
            connector = change_mean.add_feature(active_sit_new_mean, effect=True)
            active_sit_new.sign.add_out_meaning(connector)
            sit_to_plan = active_pm.sign.images[1]
        else:
            #create Abstract action and rotate in while planning backward
            change_name = 'Abstract'
            active_sit_new_mean = active_sit_new.copy('image', 'meaning')
            change_mean = self.world_model[change_name].add_meaning()
            connector = change_mean.add_feature(active_sit_new_mean)
            active_sit_new.sign.add_out_meaning(connector)
            connector = change_mean.add_feature(active_pm, effect=True)
            active_pm.sign.add_out_meaning(connector)
            sit_to_plan = active_sit_new_mean.sign.images[1]
        if current_plan:
            current_plan.append((sit_to_plan, change_name, change_mean, self.world_model['I'],
                                 current_plan[-1][4], current_plan[-1][5], act_descr))
        else:
            orient = active_sit_new.get_iner(self.world_model['orientation'], 'image')[0]
            side = None
            for sign in orient.get_signs():
                if sign.name != agent and sign.name != 'I':
                    side = sign.name
                    break
            ag_coords = {'x':self.additions[0][0]['objects'][agent]['x'],
                         'y':self.additions[0][0]['objects'][agent]['y'],
                         'r':self.additions[0][0]['objects'][agent]['r']}
            current_plan.append(
                (sit_to_plan, change_name, change_mean, self.world_model['I'], (ag_coords, side), (active_map, self.clarification_lv),
                 act_descr))

        #start planning process
        plans = self._map_sp_iteration(active_sit_new, active_map, iteration=iteration, current_plan=current_plan, goal_map=goal_map, goal_pm=goal_sit_new)

        current_plans = None
        if plans:
            current_plans = plans
            active_pm = current_plans[0][-1][0]
            iteration += len(current_plans[0])

        return current_plans, active_pm, active_map, iteration


    def _generate_meanings(self, chains):
        def __get_role_index(chain):
            index = 0
            rev_chain = reversed(chain)
            for el in rev_chain:
                if len(el.cause) == 0:
                    continue
                elif len(el.cause) == 1:
                    if len(el.cause[0].coincidences) ==1:
                        index = chain.index(el)
                    else:
                        return index
                else:
                    return index
            return None

        def __get_big_role_index(chain):
            index = None
            for el in chain:
                if len(el.cause) == 1:
                    if len(el.cause[0].coincidences) ==1:
                        index = chain.index(el)
                        break
                else:
                    continue
            if index:
                return index
            return None

        big_replace = {}

        replace_map = {}
        main_pm = None
        for chain in chains:
            role_index = __get_role_index(chain)
            if role_index:
                if not chain[role_index].sign in replace_map:
                    replace_map[chain[role_index].sign] = [chain[-1]]
                else:
                    if not chain[-1] in replace_map[chain[role_index].sign]:
                        replace_map[chain[role_index].sign].append(chain[-1])
            role_index = __get_big_role_index(chain)
            if role_index:
                if not chain[role_index].sign in big_replace:
                    big_replace[chain[role_index].sign] = [chain[role_index + 1]]
                else:
                    if not chain[role_index + 1] in big_replace[chain[role_index].sign]:
                        big_replace[chain[role_index].sign].append(chain[role_index + 1])
                main_pm = chain[0]

        connectors = self.I_sign.out_meanings

        main_pm_len = len(main_pm.cause) + len(main_pm.effect) + 2

        mapped_actions = {}
        # for agent_con in connectors:
        #     for con in agent_con:
        #         if con.in_sign == main_pm.sign:
        #             mapped_actions.setdefault(con.out_sign, set()).add(con.in_sign.meanings[con.in_index])
        for agent_con in connectors:
            if agent_con.in_sign == main_pm.sign:
                mapped_actions.setdefault(agent_con.out_sign, set()).add(agent_con.in_sign.meanings[agent_con.in_index])

        new_map = {}
        rkeys = {el for el in replace_map.keys()}
        pms = []
        # to much acts for pick-up!
        for agent, lpm in mapped_actions.items():
            for pm in lpm.copy():
                if len(pm.cause) + len(pm.effect) != main_pm_len:
                    lpm.remove(pm)
                    pm.sign.remove_meaning(pm)
                    continue
                pm_signs = set()
                pm_mean = pm.spread_down_activity('meaning', 3)
                for pm_list in pm_mean:
                    pm_signs |= set([c.sign for c in pm_list])
                role_signs = rkeys & pm_signs
                if not role_signs:
                    lpm.remove(pm)
                    if not pms:
                        pms.append((agent, pm))
                    else:
                        for _, pmd in copy(pms):
                            if pmd.resonate('meaning', pm):
                                break
                        else:
                            pms.append((agent, pm))
            old_pms = []

            for pm in lpm:
                if len(pm.cause) + len(pm.effect) != main_pm_len:
                    continue
                pm_signs = set()
                pm_mean = pm.spread_down_activity('meaning', 3)
                for pm_list in pm_mean:
                    pm_signs |= set([c.sign for c in pm_list])
                if pm_signs not in old_pms:
                    old_pms.append(pm_signs)
                else:
                    continue
                role_signs = rkeys & pm_signs
                for role_sign in role_signs:
                    new_map[role_sign] = replace_map[role_sign]

                for chain in pm_mean:
                    if chain[-1].sign in big_replace and not chain[-1].sign in new_map :
                        for cm in big_replace.get(chain[-1].sign):
                            if self.world_model['cell?x'] in cm.get_signs() and self.world_model['cell?y'] in cm.get_signs():
                                new_map[chain[-1].sign] = [cm]

                ma_combinations = self.mix_pairs(new_map)

                for ma_combination in ma_combinations:
                    cm = pm.copy('meaning', 'meaning')
                    for role_sign, obj_pm in ma_combination.items():
                        if obj_pm.sign in pm_signs:
                            continue
                        obj_cm = obj_pm.copy('significance', 'meaning')
                        try:
                            cm.replace('meaning', role_sign, obj_cm)
                        except Exception:
                            print("In action {0} role {1} object {2}".format(pm.sign.name, role_sign.name, obj_cm.sign.name))
                    for matr in cm.spread_down_activity('meaning', 6):
                        if matr[-1].sign.name == 'cell?y' or matr[-1].sign.name == 'cell?x':
                            celly = self.world_model['cell?y']
                            cellx = self.world_model['cell?x']
                            cell_y_change = ma_combination[celly].copy('meaning', 'meaning')
                            cm.replace('meaning', celly, cell_y_change)
                            cell_x_change = ma_combination[cellx].copy('meaning', 'meaning')
                            cm.replace('meaning', cellx, cell_x_change)
                            break

                    if not pms:
                        pms.append((agent, cm))
                    else:
                        for _, pmd in copy(pms):
                            if pmd.resonate('meaning', cm):
                                break
                        else:
                            pms.append((agent, cm))
                if len(old_pms) == 64:
                    break

        return pms

    def _check_result(self, pm, result_pm):
        if len(pm.effect):
            result = True
        else:
            result = False
        for event in pm.effect:
            for fevent in result_pm.cause:
                if event.resonate('meaning', fevent, True):
                    break
            else:
                result = False
                break
        return result, pm

    def _sp_check_activity(self, active_pm, scripts, prev_pms, iteration, prev_state, prev_act):
        heuristic = []
        for agent, script in scripts:
            if agent is None: agent = self.world_model['I']
            estimation, cell_coords_new, new_x_y, \
            cell_location, near_loc, region_map, current_direction = self._state_prediction(active_pm, script, agent, iteration)
            old_cl_lv = self.clarification_lv
            counter = 0
            path = -1

            if self.subsearch == 'greedy':
                counter, path = self.greedy_search(active_pm, script, iteration, new_x_y, estimation, cell_coords_new,prev_pms, prev_state, prev_act, cell_location)
            elif self.subsearch == 'ASearch':
                counter, path =  self.ASearch(active_pm, script, iteration, new_x_y, estimation, cell_coords_new,prev_pms, prev_state, prev_act, cell_location, current_direction)

            if 'task' in script.sign.name:
                counter +=10

            if 'task' in script.sign.name and not 'sub' in script.sign.name:
                self.clarification_lv = old_cl_lv
            heuristic.append((counter, script.sign.name, script, agent, path))

        if heuristic:
            best_heuristics = max([heu[0] for heu in heuristic if heu[0] >=0])
            heus = list(filter(lambda x: x[0] == best_heuristics, heuristic))
            pl = min([el[-1] for el in heus])
            heus = list(filter(lambda x: x[-1] == pl, heus))
            return heus
        else:
            return None

    def difference(self, active, estim):
        old = active - estim
        new = estim - active
        eq = False
        if len(old) != len(new):
            eq = True
        return eq, old, new

    def cell_closer(self, curcell, fcell, agent):
        if self.backward:
            goal = self.init_state
        else:
            goal = self.goal_state
        ag_plce = goal['objects'][agent]['x'], goal['objects'][agent]['y']
        cur_mid = curcell[0]+(curcell[2] - curcell[0])/2 , curcell[1] + (curcell[3]-curcell[1])/2
        fcell_mid = fcell[0]+(fcell[2] - fcell[0])/2 , fcell[1] + (fcell[3]-fcell[1])/2
        if (((fcell_mid[0] - ag_plce[0]) ** 2 + (fcell_mid[1] - ag_plce[1]) ** 2) ** (0.5)) < \
                (((cur_mid[0] - ag_plce[0]) ** 2 + ((cur_mid[1] - ag_plce[1]) ** 2)) ** (0.5)):
            return True
        return False

    def linear_cell(self, curcell, fcell, agent):
        delta = 3
        cur_mid = curcell[0]+(curcell[2] - curcell[0])/2 , curcell[1] + (curcell[3]-curcell[1])/2
        fcell_mid = fcell[0]+(fcell[2] - fcell[0])/2 , fcell[1] + (fcell[3]-fcell[1])/2
        ag_plce = self.goal_state['objects'][agent]['x'], self.goal_state['objects'][agent]['y']
        if fcell_mid[0]-delta <= ag_plce[0] <= fcell_mid[0]+ delta or fcell_mid[1]-delta <= ag_plce[1] <= fcell_mid[1]+ delta:
            if (((fcell_mid[0] - ag_plce[0])**2 + (fcell_mid[1] - ag_plce[1])**2)**(0.5)) < \
                    (((cur_mid[0] - ag_plce[0])**2 + ((cur_mid[1] - ag_plce[1])**2))**(0.5)):
                return True

        return False

    # def _applicable_events(self, pm, effect = False):
    #     applicable = []
    #     if effect:
    #         search_in_part = pm.effect
    #     else:
    #         search_in_part = pm.cause
    #     for event in search_in_part:
    #         if len(event.coincidences) == 1:
    #             flag = False
    #             for connector in event.coincidences:
    #                 if connector.out_sign in self.agents:
    #                     flag = True
    #             if flag:
    #                 continue
    #         applicable.append(event)
    #     return applicable

    def recursive_files(self, direct, ext):
        import os
        extfiles = []
        for root, subfolder, files in os.walk(direct):
            for file in files:
                if file.endswith(ext):
                    extfiles.append(os.path.join(root, file))
            for sub in subfolder:
                extfiles.extend(self.recursive_files(os.path.join(root, sub), ext))
            return extfiles

    def scale_history_situation(self, history_benchmark, iteration, start, finish):
        new_objects = {}
        if self.backward:
            start, finish = finish, start
        for stobj, scoords in history_benchmark[start]['objects'].items():
            for curobj, curcoords in self.additions[0][iteration]['objects'].items():
                if stobj == curobj:
                    koef_x = curcoords['x'] - scoords['x']
                    koef_y = curcoords['y'] - scoords['y']
                    new_objects[curobj] = {'x': history_benchmark[finish]['objects'][stobj]['x']+koef_x,
                                           'y': history_benchmark[finish]['objects'][stobj]['y']+koef_y,
                                           'r': history_benchmark[finish]['objects'][stobj]['r']}

        return {'objects': new_objects}, {'map-size': self.additions[3]['map-size'], 'wall': self.additions[3]['wall']}


    def history_action(self, active_pm, script, agent, iteration):
        import os
        import pkg_resources
        import json
        benchmark = None
        history_benchmark = None
        paths = []
        delim = '/'
        logging.info("Searching benchmark for %s experience action..." % script.sign.name)
        for name in os.listdir(''):
            if 'benchmark' in name.lower():
                paths.append('.' + delim +name + delim)
        if not paths:
            if not benchmark:
                modules = ['mapplanner', 'mapspatial']
                for module in modules:
                    try:
                        path = pkg_resources.resource_filename(module, 'benchmarks')
                    except ModuleNotFoundError:
                        logging.info("Module %s is not loaded. Searching further..." % module)
                        continue
                    logging.info("Benchmarks was found in %s module." % module)
                    paths.append(path)
                    break
        for direct in paths:
            files = self.recursive_files(direct, '.json')
            for file in files:
                with open(file) as data:
                    jfile = json.load(data)
                    if 'task-name' in jfile:
                        if script.sign.name.endswith(jfile['task-name']):
                            logging.info("Path to benchmark is: %s" % file)
                            history_benchmark = jfile
                            break
                        else:
                            continue
                    else:
                        continue

        start = 'global-start'
        finish = 'global-finish'

        try:
            a = history_benchmark[start]
        except KeyError:
            start = 'start'
            finish = 'finish'


        parsed, static = self.scale_history_situation(history_benchmark, iteration, start, finish)

        region_map, cell_map, cell_location, near_loc, cell_coords, _, _ = signs_markup(parsed, static, 'agent?ag')
        events = []
        for ev in active_pm.cause:
            if len(ev.coincidences) == 1:
                for con in ev.coincidences:
                    if con.out_sign.name == "I":
                        events.append(ev)

        orientation = history_benchmark[finish][self.I_obj.name]['orientation']
        direction = self.world_model[orientation]

        history_benchmark[finish]['objects'].update(parsed['objects'])
        new_x_y = history_benchmark[finish]
        new_x_y['map-size'] = self.additions[3]['map-size']
        # new_x_y['wall'] = self.additions[3]['wall']

        agent_state = state_prediction(agent, history_benchmark[finish], self.world_model)

        sit_name = st.SIT_PREFIX + str(st.SIT_COUNTER)
        st.SIT_COUNTER+=1
        estimation = define_situation(sit_name + 'sp', cell_map, events, agent_state, self.world_model)
        state_fixation(estimation, cell_coords, self.world_model, 'cell')

        #check size
        old_size = self.world_model['exp_*map*'].images[2].spread_down_activity_view(depth=1)
        new_size = self.world_model['*map*'].images[2].spread_down_activity_view(depth=1)

        if old_size != new_size:
            #Calculate how many times cur sit bigger than old or vice versa
            self.clarification_lv += int((new_size['region-8'][3] // old_size['region-8'][3] ) / 3)

        return cell_map, direction, estimation, cell_coords, new_x_y, cell_location, \
            near_loc, region_map

    def new_action(self, active_pm, script, agent, iteration):
        direction = None
        cell = None
        block_name = None
        ground_block = None

        fast_estimation, events, holding = self._time_shift_spat(active_pm, script)
        searched = search_cm(fast_estimation, [self.world_model['orientation'], self.world_model['holding'], self.world_model['employment']] )
        employment = searched[self.world_model['employment']][0]

        if holding:
            holding = holding[0]
        orientation = searched[self.world_model['orientation']][0]
        for sign in orientation.get_signs():
            if sign != agent and sign != self.world_model['I']:
                direction = sign
                break
        if script.sign.name == 'move':
            if not self.backward:
                for sign in employment.get_signs():
                    if sign != agent:
                        cell = sign.name
                        break
            else:
                # find mirror cell
                goal_reg = [reg for reg, place in self.additions[1]['region-4'].items() if place[1] == direction.name][0]
                mirror_side = [place[1] for reg, place in self.additions[1][goal_reg].items() if reg == 'region-4'][0]
                mirror_reg = [reg for reg, place in self.additions[1]['region-4'].items() if place[1] == mirror_side][0]
                cell = 'cell-' + mirror_reg.split('-')[-1]
        else:
            for sign in employment.get_signs():
                if sign != agent:
                    cell = sign.name
                    break
        agent_state = state_prediction(agent, direction, self.world_model, holding)

        cell_coords = active_pm.sign.images[2].spread_down_activity_view(depth = 1)[cell]

        new_x_y = deepcopy(self.additions[0][iteration])
        if script.sign.name == 'move':
            ag_x = new_x_y['objects'][self.I_obj.name]['x']
            ag_y = new_x_y['objects'][self.I_obj.name]['y']
            if direction.name == 'below':
                new_x_y['objects'][self.I_obj.name]['y'] = ag_y + (cell_coords[3] - cell_coords[1])
            elif direction.name == 'above':
                new_x_y['objects'][self.I_obj.name]['y'] = ag_y - (cell_coords[3] - cell_coords[1])
            elif direction.name == 'left':
                new_x_y['objects'][self.I_obj.name]['x'] = ag_x - (cell_coords[2] - cell_coords[0])
            elif direction.name == 'right':
                new_x_y['objects'][self.I_obj.name]['x'] = ag_x + (cell_coords[2] - cell_coords[0])
            elif direction.name == 'above-right':
                new_x_y['objects'][self.I_obj.name]['x'] = ag_x + (cell_coords[2] - cell_coords[0])
                new_x_y['objects'][self.I_obj.name]['y'] = ag_y - (cell_coords[3] - cell_coords[1])
            elif direction.name == 'above-left':
                new_x_y['objects'][self.I_obj.name]['x'] = ag_x - (cell_coords[2] - cell_coords[0])
                new_x_y['objects'][self.I_obj.name]['y'] = ag_y - (cell_coords[3] - cell_coords[1])
            elif direction.name == 'below-left':
                new_x_y['objects'][self.I_obj.name]['x'] = ag_x - (cell_coords[2] - cell_coords[0])
                new_x_y['objects'][self.I_obj.name]['y'] = ag_y + (cell_coords[3] - cell_coords[1])
            elif direction.name == 'below-right':
                new_x_y['objects'][self.I_obj.name]['x'] = ag_x + (cell_coords[2] - cell_coords[0])
                new_x_y['objects'][self.I_obj.name]['y'] = ag_y + (cell_coords[3] - cell_coords[1])
        if script.sign.name == 'rotate':
            new_x_y[self.I_obj.name]['orientation'] = direction.name

        if script.sign.name != 'rotate' and script.sign.name != 'move':
            block_name = [sign.name for sign in script.get_iner(self.world_model['holding'], 'meaning')[0].get_signs() if
                          'block' in sign.name][0]

        # for pick-up script
        if script.sign.name == 'pick-up':
            if block_name in new_x_y['objects'].keys():
                new_x_y['objects'].pop(block_name)
            new_x_y[self.I_obj.name].pop('handempty')
            new_x_y[self.I_obj.name]['holding'] = {'cause': [self.I_obj.name, block_name], 'effect': []}

        # for put-down script
        elif script.sign.name == 'put-down':
            new_x_y['objects'][block_name] = {'x': new_x_y['objects'][self.I_obj.name]['x'],
                                              'y': new_x_y['objects'][self.I_obj.name]['y'] + new_x_y[self.I_obj.name][
                                                  'activity'],
                                              'r': self.additions[0][0]['objects'][block_name]['r']}
            new_x_y[self.I_obj.name].pop('holding')
            new_x_y[self.I_obj.name]['handempty'] = {'cause': [], 'effect': []}


        elif script.sign.name == 'stack':
            for el in script.cause:
                el_signs = el.get_signs()
                if 'clear' in [sign.name for sign in el_signs]:
                    ground_block = [sign for sign in el_signs if sign.name != 'clear'][0]
                    break
            ground_coords = self.additions[0][iteration]['objects'][ground_block.name]['x'], \
                            self.additions[0][iteration]['objects'][ground_block.name]['y'],
            new_x_y['objects'][block_name] = {'x': ground_coords[0],
                                              'y': ground_coords[1],
                                              'r': self.additions[0][0]['objects'][block_name]['r']}
            new_x_y[self.I_obj.name].pop('holding')
            new_x_y[self.I_obj.name]['handempty'] = {'cause': [], 'effect': []}

        elif script.sign.name == 'unstack':
            if block_name in new_x_y['objects'].keys():
                new_x_y['objects'].pop(block_name)
            new_x_y[self.I_obj.name].pop('handempty')
            new_x_y[self.I_obj.name]['holding'] = {'cause': [self.I_obj.name, block_name], 'effect': []}

        region_map, cell_map, cell_location, near_loc, cell_coords_new, _,_ = signs_markup(new_x_y, self.additions[3], self.I_obj.name, cell_coords)
        sit_name = st.SIT_PREFIX + str(st.SIT_COUNTER)
        st.SIT_COUNTER+=1

        if script.sign.name == 'rotate' or  script.sign.name == 'move':
            conditions = new_x_y['conditions']
        else:
            from mapspatial.agent.planning_agent import get_conditions
            conditions = get_conditions(new_x_y, (0, script.sign.name), block_name, ground_block)
            new_x_y['conditions'] = conditions

        estimation = define_situation(sit_name + 'sp', cell_map, conditions, agent_state, self.world_model)
        estimation = update_situation(estimation, cell_map, self.world_model, fast_estimation)
        state_fixation(estimation, cell_coords_new, self.world_model, 'cell')

        return cell_map, direction, estimation, cell_coords_new, new_x_y, cell_location, near_loc, region_map

    def sub_action(self, active_pm, script, agent, iteration):
        applicable = {'st':None, 'fn':None}
        for sit in self.exp_sits:
            if sit.sign.meanings:
                sit_mean = sit.sign.meanings[1]
            else:
                sit_mean = sit.copy('image', 'meaning')
            if applicable['fn'] is None:
                result, checked = self._check_result(script, sit_mean)
                if result:
                    applicable['fn'] = sit_mean
            if applicable['st'] is None:
                result, checked = self._check_activity(script, sit_mean)
                if result:
                    applicable['st'] = sit_mean
            if applicable['fn'] and applicable['st']:
                break
        else:
            if applicable['st'] is None:
                logging.warning('Lost start situation for %s'%script.sign.name)
            elif applicable['fn'] is None:
                logging.warning('Lost finish situation for %s'%script.sign.name)
        exp_start = applicable['st']
        exp_finish = applicable['fn']
        if exp_finish.sign.images:
            exp_finish = exp_finish.sign.images[1]

        events, direction, holding = pm_parser(exp_finish, agent.name, self.world_model, base='image')

        new_x_y = deepcopy(self.additions[0][iteration])
        new_x_y[self.I_obj.name]['orientation'] = direction.name
        cell_coords_st = exp_start.sign.images[2].spread_down_activity_view(depth=1)
        cell_coords_fn = exp_finish.sign.images[2].spread_down_activity_view(depth=1)

        st_cell_4 = cell_coords_st['cell-4']
        fn_cell_4 = cell_coords_fn['cell-4']

        if ((st_cell_4[3] - st_cell_4[1]) != (fn_cell_4[3] - fn_cell_4[1])) and ((st_cell_4[2] - st_cell_4[0]) != (fn_cell_4[2] - fn_cell_4[0])):
            new_x_y['objects'][self.I_obj.name]['x'], new_x_y['objects'][self.I_obj.name]['y'] = self.adapt_exp_situation(applicable, active_pm)
            # old_x = (fn_cell_4[2] - fn_cell_4[0]) // 2
            # old_y = (fn_cell_4[3] - fn_cell_4[1]) // 2
            # cell_coords_new = [new_x_y['objects'][self.I_obj.name]['x'] - old_x, new_x_y['objects'][self.I_obj.name]['y']-old_y,
            #                    new_x_y['objects'][self.I_obj.name]['x'] + old_x, new_x_y['objects'][self.I_obj.name]['y']+old_y]

            if holding:
                block_name = [sign.name for sign in holding.get_signs() if 'block' in sign.name][0]
                if block_name in new_x_y['objects'].keys():
                    new_x_y['objects'].pop(block_name)
                if 'handempty' in new_x_y[self.I_obj.name].keys():
                    new_x_y[self.I_obj.name].pop('handempty')
                new_x_y[self.I_obj.name]['holding'] = {'cause': [self.I_obj.name, block_name], 'effect': []}
            else:
                # stack script here
                if 'holding' in new_x_y[self.I_obj.name]:
                    block_name = [el for el in new_x_y[self.I_obj.name]['holding']['cause'] if el != self.I_obj.name][0]
                    ground_block = None
                    on_signs = search_cm([ev for ev in script.effect], [self.world_model['on']], base = 'meaning')
                    flag = False
                    for ons in on_signs.values():
                        for elem in ons:
                            cm_signs_names  = [s.name for s in elem.get_signs()]
                            if block_name in cm_signs_names:
                                for block in cm_signs_names:
                                    if block in new_x_y['objects']:
                                        ground_block = block
                                        flag = True
                                        break
                            if flag:
                                break
                        if flag:
                            break
                    if self.additions[0]:
                        block_radius = self.additions[0][0]['objects'][block_name]['r']
                    else:
                        block_radius = 1
                    new_x_y['objects'][block_name] = new_x_y['objects'][ground_block]
                    new_x_y['objects'][block_name]['r'] = block_radius
                    new_x_y[self.I_obj.name].pop('holding')
                    new_x_y[self.I_obj.name]['handempty'] = {'cause':[], 'effect':[]}

            region_map, cell_map, cell_location, near_loc, cell_coords, _, _ = signs_markup(new_x_y,self.additions[3],
                                                                                                self.I_obj.name, fn_cell_4)

            agent_state = state_prediction(agent, direction, self.world_model, holding = holding)
            sit_name = st.SIT_PREFIX + str(st.SIT_COUNTER)
            st.SIT_COUNTER += 1
            new_situation = define_situation(sit_name + 'sp', cell_map, events, agent_state, self.world_model)

            state_fixation(new_situation, cell_coords, self.world_model, 'cell')

            return cell_map, direction, new_situation, cell_coords, new_x_y, cell_location, \
            near_loc, region_map
        else:

            new_x_y['objects'][self.I_obj.name]['x'] = (fn_cell_4[2] - fn_cell_4[0]) // 2 + fn_cell_4[0]
            new_x_y['objects'][self.I_obj.name]['y'] = (fn_cell_4[3] - fn_cell_4[1]) // 2 + fn_cell_4[1]

            region_map, cell_map, cell_location, near_loc, _, _, _ = signs_markup(new_x_y, self.additions[3],
                                                                                                self.I_obj.name, fn_cell_4)
            state_fixation(exp_finish, cell_coords_fn, self.world_model, 'cell')

            return cell_map, direction, exp_finish, cell_coords_fn, new_x_y, cell_location, \
            near_loc, region_map

    def adapt_exp_situation(self, applicable, active_pm):
        """
        This function allows to adapt old exp to new task. Already used, when we adapt exp
        200x200 matrix to 600x600 matrix.
        :param applicable: start and finish sits
        :param active_pm: current pm on LS
        :return: adapted agent coords
        """
        cell_coords_current = active_pm.sign.images[2].spread_down_activity_view(depth=1)
        ag_coords_current_x = (cell_coords_current['cell-4'][2] - cell_coords_current['cell-4'][0]) // 2 + cell_coords_current['cell-4'][0]
        ag_coords_current_y = (cell_coords_current['cell-4'][3] - cell_coords_current['cell-4'][1]) // 2 + cell_coords_current['cell-4'][1]

        cell_coords_old_s = applicable['st'].sign.images[2].spread_down_activity_view(depth=1)
        ag_coords_old_x_s = (cell_coords_old_s['cell-4'][2] - cell_coords_old_s['cell-4'][0]) // 2 + \
                          cell_coords_old_s['cell-4'][0]
        ag_coords_old_y_s = (cell_coords_old_s['cell-4'][3] - cell_coords_old_s['cell-4'][1]) // 2 + \
                          cell_coords_old_s['cell-4'][1]

        dif_x = ag_coords_current_x - ag_coords_old_x_s
        dif_y = ag_coords_current_y - ag_coords_old_y_s

        cell_coords_old_f = applicable['fn'].sign.images[2].spread_down_activity_view(depth=1)
        ag_coords_old_x_f = (cell_coords_old_f['cell-4'][2] - cell_coords_old_f['cell-4'][0]) // 2 + \
                          cell_coords_old_f['cell-4'][0]
        ag_coords_old_y_f = (cell_coords_old_f['cell-4'][3] - cell_coords_old_f['cell-4'][1]) // 2 + \
                          cell_coords_old_f['cell-4'][1]

        return dif_x+ag_coords_old_x_f, dif_y+ag_coords_old_y_f

    def _state_prediction(self, active_pm, script, agent, iteration, flag=False):
        if script.sign.images and 'task' in script.sign.name and not 'sub' in script.sign.name:
            cell_map, direction, estimation, cell_coords_new, new_x_y, cell_location, \
            near_loc, region_map = self.history_action(active_pm, script, agent, iteration)
        elif script.sign.images and 'sub' in script.sign.name:
            cell_map, direction, estimation, cell_coords_new, new_x_y, cell_location, \
            near_loc, region_map = self.sub_action(active_pm, script, agent, iteration)
        else:
            cell_map, direction, estimation, cell_coords_new, new_x_y, cell_location, \
            near_loc, region_map = self.new_action(active_pm, script, agent, iteration)

        if flag:
            region = None
            for reg, cellz in cell_location.items():
                if 'cell-4' in cellz:
                    region = reg
                    break
            self.additions[2]["I"][iteration+1] = cell_map
            self.additions[0][max(self.additions[0])]['cl_lv'] = self.clarification_lv
            #if self.backward:
                #goal_reg = [reg for reg, place in self.additions[1]['region-4'].items() if place[1] == direction.name][0]
                #direct = [place[1] for reg, place in self.additions[1][goal_reg].items() if reg == 'region-4'][0]
            print("{0}. act: {1}, cell: {2}, dir: {3}, reg: {4}".format(str(iteration), script.sign.name, cell_coords_new['cell-4'],
                                                                   direction.name, region))
            return estimation, cell_coords_new, new_x_y, cell_location, near_loc, region_map, direction.name

        return estimation, cell_coords_new, new_x_y, cell_location, near_loc, region_map, direction

    def _step_generating(self, active_pm, active_map, script, agent, iteration, prev_state, param):
        next_pm, cell_coords, parsed_map, cell_location, \
            near_loc, region_map, direction = self._state_prediction(active_pm, script, agent, iteration, param)
        prev_state.append(cell_coords['cell-4'])
        state_fixation(next_pm, cell_coords, self.world_model, 'cell')
        parsed_map['cl_lv'] = self.clarification_lv
        self.additions[0][iteration + 1] = parsed_map
        if self.change_map(active_map, cell_location):
            active_map = define_map(st.MAP_PREFIX + str(st.SIT_COUNTER), region_map, cell_location, near_loc, self.additions[1],
                                 self.world_model)
            logging.info('Карта пересчитана!')
        elif iteration > 0:
            if list(self.additions[2]["I"][iteration].values()) != list(self.additions[2]["I"][iteration - 1].values()):
                active_map = define_map(st.MAP_PREFIX + str(st.SIT_COUNTER), region_map, cell_location, near_loc,
                                     self.additions[1], self.world_model)
            elif self.clarification_lv > 0:
                active_map = define_map(st.MAP_PREFIX + str(st.SIT_COUNTER), region_map, cell_location, near_loc,
                                        self.additions[1], self.world_model)
        # elif 'exp_*map*' in self.world_model:
        #     old_size = self.world_model['exp_*map*'].images[2].spread_down_activity_view(depth=1)
        #     new_size = self.world_model['*map*'].images[2].spread_down_activity_view(depth=1)
        #     if old_size != new_size:
        #         active_map = define_map(st.MAP_PREFIX + str(st.SIT_COUNTER), region_map, cell_location, near_loc,
        #                                 self.additions[1],
        #                                 self.world_model)
        #         logging.info('Карта пересчитана!')
        return next_pm, active_map.sign.images[1], prev_state, direction

    def _time_shift_spat(self, active_pm, script):
        script_img = script.copy('meaning', 'image')
        pm_events = []
        events = []
        if self.backward:
            attr1 = 'effect'
            attr2 = 'cause'
        else:
            attr1 = 'cause'
            attr2 = 'effect'
        if not self.backward:
            for event in active_pm.sign.images[1].cause:
                for es in getattr(script_img, attr1):
                    if event.resonate('image', es):
                        break
                else:
                    pm_events.append(event)
        for event in getattr(script_img, attr2):
            pm_events.append(event)

        searched = search_cm(pm_events, [self.world_model['holding']])
        holding = searched[self.world_model['holding']]

        for ev in itertools.chain(getattr(script, attr2), active_pm.cause):
            if len(ev.coincidences) == 1:
                for con in ev.coincidences:
                    if con.out_sign.name == "I":
                        for event in copy(events):
                            if event.resonate('meaning', ev):
                                break
                        else:
                            events.append(ev)
            elif not holding:
                if "I" in [s.name for s in ev.get_signs()]:
                    events.append(ev)

        return pm_events, events, holding

    def change_map(self, active_map, cell_location):
        pms = active_map.spread_down_activity('meaning', 4)
        pm_list = []
        contain_reg = None
        for location, cells in cell_location.items():
            if 'cell-4' in cells:
                contain_reg = location
                break
        for iner in pms:
            iner_names = [s.sign.name for s in iner]
            if 'include' in iner_names:
                pm_list.append(iner[-1])
        for pm in pm_list:
            if pm.sign.name != 'cell-4':
                if pm.sign.name == contain_reg:
                    return False
        return True

    def get_stright(self, estimation, dir_sign):
        es = estimation.spread_down_activity('meaning', 4)
        grouped = {}
        for key, group in itertools.groupby(es, lambda x: x[1]):
            for pred in group:
                grouped.setdefault(key, []).append(pred[-1])
        stright_cell = None
        used_key = None
        for key, item in grouped.items():
            it_signs = {am.sign for am in item}
            if dir_sign in it_signs:
                stright_cell = [sign for sign in it_signs if
                                sign.name != 'cell-4' and sign != dir_sign and sign.name != "I"]
                if stright_cell:
                    stright_cell = stright_cell[0]
                    used_key = key
                    break
        for key, item in grouped.items():
            if key != used_key:
                it_signs_names = {am.sign.name for am in item}
                if stright_cell.name in it_signs_names:
                    if 'nothing' in it_signs_names:
                        return stright_cell, None
                    else:
                        items = [it for it in item if it.sign != stright_cell and it.sign.name != 'cell-4']
                        if items: return stright_cell, items

    @staticmethod
    def mix_pairs(replace_map):
        new_chain = {}
        elements = []
        merged_chains = []
        used_roles = []
        replace_map = list(replace_map.items())

        def get_role(obj, roles):
            for role in roles:
                if obj in role[1]:
                    return role

        for item in replace_map:
            elements.append(item[1])
        elements = list(itertools.product(*elements))
        clean_el = copy(elements)
        for element in clean_el:
            if not len(set(element)) == len(element):
                elements.remove(element)
        for element in elements:
            for obj in element:
                avalaible_roles = [x for x in replace_map if x not in used_roles]
                role = get_role(obj, avalaible_roles)
                if role:
                    used_roles.append(role)
                    new_chain[role[0]] = obj
            merged_chains.append(new_chain)
            new_chain = {}
            used_roles = []
        return merged_chains

    def __get_tactical(self, counter, script, cell_coords, new_x_y, active_pm):
        new_cell = cell_coords['cell-4']
        size = new_cell[2] - new_cell[0], new_cell[3] - new_cell[1]
        old_orientation = self.additions[0][max(self.additions[0].keys())][self.I_obj.name]['orientation']
        new_orientation = new_x_y[self.I_obj.name]['orientation']

        agent_old = deepcopy(new_x_y['objects'][self.I_obj.name])
        if script.sign.name == 'move':
            old_cell = active_pm.sign.images[2].spread_down_activity_view(1)['cell-4']
            ag_c = old_cell[0] + ((old_cell[2]-old_cell[0])/2), old_cell[1] + ((old_cell[3]-old_cell[1])/2)
            agent_old['x'] = ag_c[0]
            agent_old['y'] = ag_c[1]
        agent_new = new_x_y['objects'][self.I_obj.name]

        start = {'agent-orientation': old_orientation}
        start[self.I_obj.name] = agent_old
        finish = {'agent-orientation': new_orientation}
        finish[self.I_obj.name] = agent_new

        request = {}
        request['start'] = start
        request['finish'] = finish
        request['cell-size'] = size
        request['name'] = script.sign.name
        request['counter'] = counter

        with open(self.task_file) as data_file1:
            new_request = json.load(data_file1)

        new_request['current-action'] = request

        import platform
        import os
        if platform.system == 'Linux':
            delim = '/'
            path = '/'
            for part in self.task_file.split('/')[1:-1]:
                path += part + '/'
        else:
            delim = '\\'
            path = ''
            for part in self.task_file.split(delim)[:-1]:
                path += part + delim

        request_path = path + 'requests'+delim+'request_' + script.sign.name + '_' + str(counter)+ '.json'

        with open(request_path, 'w') as outfile:
            json.dump(new_request, outfile)

        #TODO Here is a server
        exepath = ''
        for part in self.task_file.split(delim)[:-4]:
            exepath += part+delim
        exepath+= 'astar' +delim+ 'ASearch.exe'

        cmd = [exepath, request_path]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            logging.info(stderr)
            logging.info(p.returncode)
            p2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p2.returncode != 0:
                raise Exception('Can not access the Astar search')
            else:
                subprocess.Popen.kill(p2)

        else:
            subprocess.Popen.kill(p)

        response_path = path + 'responses'+delim+'result_' + script.sign.name + '_' + str(counter)+ '.json'
        with open(response_path) as data_file1:
            tactical_response = json.load(data_file1)

        tactical_response = tactical_response['result']
        return tactical_response

    def ASearch(self, active_pm, script, iteration, new_x_y, estimation, cell_coords_new,prev_pms, prev_state, prev_act, cell_location, current_direction):
        counter = 0
        path = 0
        if not 'task' in script.sign.name:
            stright = self.get_stright(active_pm, current_direction)
        else:
            stright = self.get_stright(estimation, current_direction)
        tactical_response = self.__get_tactical(iteration, script, cell_coords_new, new_x_y, active_pm)
        t_c = tactical_response['target-cell']
        # Coords of the local goal cell
        targ_coord = t_c[0] + ((t_c[2] - t_c[0]) // 2), t_c[1] + ((t_c[3] - t_c[1]) // 2)
        # Coords of the current cell
        cur_c = active_pm.sign.images[2].spread_down_activity_view(1)['cell-4']
        cur_coords = cur_c[0] + ((cur_c[2] - cur_c[0]) // 2), cur_c[1] + (
                (cur_c[3] - cur_c[1]) // 2)
        # Coords of the stright cell. If there are empty space.
        if not stright[1]:
            str_c = active_pm.sign.images[2].spread_down_activity_view(1)[stright[0].name]
            strcell_coord = str_c[0] + ((str_c[2] - str_c[0]) // 2), str_c[1] + (
                    (str_c[3] - str_c[1]) // 2)
        else:
            strcell_coord = None
        if script.sign.name == 'move':
            if not tactical_response['doable']:
                logging.info('Перемещение запрещено тактическим уровнем!')
                return 0, 0
            else:
                path = math.sqrt(
                    (targ_coord[1] - strcell_coord[1]) ** 2 + (targ_coord[0] - strcell_coord[0]) ** 2)
        elif script.sign.name == 'rotate':
            if not stright[1]:
                a = math.sqrt(
                    (targ_coord[1] - cur_coords[1]) ** 2 + (targ_coord[0] - cur_coords[0]) ** 2)
                b = math.sqrt(
                    (targ_coord[1] - strcell_coord[1]) ** 2 + (targ_coord[0] - strcell_coord[0]) ** 2)

                if a > b:
                    counter += 3
                    path = b

        if 'task' in script.sign.name and not 'sub' in script.sign.name:
            self.clarification_lv = self.goal_state['cl_lv']

        if not new_x_y['objects'][self.I_obj.name]['x'] in range(0, self.additions[3]['map-size'][0]) or \
                not new_x_y['objects'][self.I_obj.name]['y'] in range(0, self.additions[3]['map-size'][1]):
            return 0, 0

        for prev in prev_pms:
            if estimation.resonate('meaning', prev, False, False):
                if cell_coords_new['cell-4'] in prev_state and self.clarification_lv == 0:
                    break
        else:
            cont_region = None
            goal_region = None
            for reg, cellz in cell_location.items():
                if 'cell-4' in cellz:
                    cont_region = reg
                    break
            agent_sign = self.world_model[self.I_obj.name]
            for iner in self.goal_map.get_iner(self.world_model['contain'], 'meaning'):
                iner_signs = iner.get_signs()
                if agent_sign in iner_signs:
                    for sign in iner_signs:
                        if sign != agent_sign and 'region' in sign.name:
                            goal_region = sign
                            break
                if goal_region:
                    break
            if goal_region.name != cont_region:

                # for move action
                if cell_coords_new['cell-4'] != active_pm.sign.images[2].spread_down_activity_view(1)['cell-4']:
                    if not stright[1]:
                        counter += 3
                        if prev_act == 'rotate':
                            counter += 4  # was +2
                        elif prev_act is None:
                            counter += 1
                # for pick-up and put-down actions
                elif self.difference(active_pm, estimation)[0]:
                    old = self.difference(active_pm, estimation)[1]
                    for event1 in old:
                        for event2 in self.check_pm.cause:
                            if event1.resonate('meaning', event2):
                                break
                        else:
                            counter += 3

            else:
                if prev_state:
                    prev_mid = prev_state[-1][0] + ((prev_state[-1][2] - prev_state[-1][0]) // 2), prev_state[-1][1] + (
                            (prev_state[-1][3] - prev_state[-1][1]) // 2)
                    goal_coords = self.goal_state['objects'][self.I_obj.name]['x'], self.goal_state['objects'][self.I_obj.name]['y']

                    a = math.sqrt(
                        (goal_coords[1] - prev_mid[1]) ** 2 + (goal_coords[0] - prev_mid[0]) ** 2)

                    # if we are already in the goal cell
                    if a < 10:
                        if self.clarification_lv <= self.goal_state['cl_lv']:
                            est_events = [event for event in estimation.cause if "I" not in event.get_signs_names()]
                            ce_events = [event for event in self.check_pm.cause if "I" not in event.get_signs_names()]
                            for event in est_events:
                                for ce in ce_events:
                                    if event.resonate('meaning', ce):
                                        counter += 1
                                        break
                    elif not stright[1]:
                        if script.sign.name == 'move':
                            counter += 2
                        a = math.sqrt(
                            (targ_coord[1] - cur_coords[1]) ** 2 + (targ_coord[0] - cur_coords[0]) ** 2)
                        b = math.sqrt(
                            (targ_coord[1] - strcell_coord[1]) ** 2 + (targ_coord[0] - strcell_coord[0]) ** 2)

                        if a >= b:
                            counter += 4
                            path = b
            return counter, path

    def greedy_search(self, active_pm, script, iteration, new_x_y, estimation, cell_coords_new, prev_pms, prev_state, prev_act, cell_location):
        counter = 0
        path = 0
        if not new_x_y['objects'][self.I_obj.name]['x'] in range(0, self.additions[3]['map-size'][0]) or \
                not new_x_y['objects'][self.I_obj.name]['y'] in range(0, self.additions[3]['map-size'][1]):
            return 0, 0

        for prev in prev_pms:
            if estimation.resonate('image', prev, False, False):
                if cell_coords_new['cell-4'] in prev_state and self.clarification_lv == 0:
                    break
        else:
            cont_region = None
            goal_region = None

            ag_orient = estimation.get_iner(self.world_model['orientation'], 'image')[0]
            iner_signs = ag_orient.get_signs()
            current_direction = None
            for sign in iner_signs:
                if sign != self.world_model["I"]:
                    current_direction = sign
                if current_direction:
                    break
            if self.backward:
                goal_reg = [reg for reg, place in self.additions[1]['region-4'].items() if place[1] == current_direction.name][0]
                mirror_side = [place[1] for reg, place in self.additions[1][goal_reg].items() if reg == 'region-4'][0]
                stright = self.get_stright(active_pm, self.world_model[mirror_side])
                current_direction = self.world_model[mirror_side]
            else:
                stright = self.get_stright(active_pm, current_direction)

            for reg, cellz in cell_location.items():
                if script.sign.name == 'rotate':
                    if stright[0].name in cellz:
                        cont_region = reg
                        break
                else:
                    if 'cell-4' in cellz:
                        cont_region = reg
                        break
            if cont_region == 'wall':
                return 0,0

            for iner in self.goal_map.get_iner(self.world_model['contain'], 'image'):
                iner_signs = iner.get_signs()
                if self.I_obj in iner_signs:
                    for sign in iner_signs:
                        if sign != self.I_obj and 'region' in sign.name:
                            goal_region = sign
                            break
                if goal_region:
                    break

            if not self.backward:
                goal = self.goal_state
            else:
                goal = self.init_state
            goal_coords = goal['objects'][self.I_obj.name]['x'], goal['objects'][self.I_obj.name]['y']

            # further are coefficient game
            if stright and script.sign.name == 'rotate':
                f_cs = cell_coords_new[stright[0].name]
            else:
                if self.backward:
                    f_cs = cell_coords_new['cell-4']
                else:
                    f_cs = cell_coords_new[stright[0].name]
            f_c = f_cs[0] + ((f_cs[2] - f_cs[0]) // 2), f_cs[1] + (
                            (f_cs[3] - f_cs[1]) // 2)
            # path next cell - goal
            path = math.sqrt(
                        (goal_coords[1] - f_c[1]) ** 2 + (goal_coords[0] - f_c[0]) ** 2)
            if prev_state:
                prev_mid = prev_state[-1][0] + ((prev_state[-1][2] - prev_state[-1][0]) // 2), prev_state[-1][1] + (
                        (prev_state[-1][3] - prev_state[-1][1]) // 2)
                size = prev_state[-1][2] - prev_state[-1][0]
            else:
                prev_mid =self.additions[0][iteration]['objects'][self.I_obj.name]['x'], self.additions[0][iteration]['objects'][self.I_obj.name]['y']
                size = f_cs[2] - f_cs[0]
            # path current cell - goal
            a = math.sqrt(
                (goal_coords[1] - prev_mid[1]) ** 2 + (goal_coords[0] - prev_mid[0]) ** 2)

            # if next cell closer - good
            if not stright[1]:
                if path <= a:
                    counter += 4

            if goal_region.name != cont_region:
                goal_dir = self.additions[1][cont_region][goal_region.name][1]
                # do not rotate to the wall if there are no hole
                if current_direction.name == goal_dir and script.sign.name == 'rotate':
                    if stright[1] and not self.clarification_lv < self.goal_cl_lv:
                        counter = 0
                    else:
                        counter += 2  # +2 if current dir is the same to goal dir
                # for move action
                elif cell_coords_new['cell-4'] != active_pm.sign.images[2].spread_down_activity_view(1)['cell-4'] and script.sign.name == 'move':
                    if not stright[1]:
                        counter += 2  # +1
                        if prev_act == 'rotate':
                            counter += 4
                        elif prev_act is None or prev_act == 'Abstract' or prev_act == 'Clarify':
                            counter += 1
                # for pick-up and put-down actions
                elif self.difference(active_pm.sign.images[1], estimation)[0] and script.sign.name != "rotate" and script.sign.name != "move":
                    old = self.difference(active_pm.sign.images[1], estimation)[1]
                    for event1 in old:
                        for event2 in self.goal_pm.sign.images[1].cause:
                            if event1.resonate('image', event2):
                                break
                        else:
                            counter += 3
                else:
                    # check closely to goal region regions
                    closely_goal = [reg for reg, ratio in self.additions[1][goal_region.name].items() if
                                    ratio[0] == 'closely']
                    closely_dirs = set()
                    if cont_region not in closely_goal:
                        for region in closely_goal:
                            closely_dirs.add(self.additions[1][cont_region][region][1])
                        if current_direction.name in closely_dirs:
                            if stright[1]:
                                counter = 0
                            else:
                                counter += 2  # +2 if rotate to closely to goal region
                    else:
                        if current_direction.name == goal_dir:
                            counter += 1  # +1 if in closely reg and rotate to future_reg
                        elif not stright[1]:
                            if self.cell_closer(cell_coords_new['cell-4'], cell_coords_new[stright[0].name],
                                                self.I_obj.name):
                                counter += 1

                if self.linear_cell(cell_coords_new['cell-4'], cell_coords_new[stright[0].name], self.I_obj.name) and not self.backward:
                    if not stright[1]:
                        counter += 2  # +2 if agent go back to the stright goal way #TODO rework when go from far

            else:
                if not stright[1] and script.sign.name == 'move':
                    counter += 1
                elif stright[1] and script.sign.name == 'rotate':
                    # rotate to blocks
                    if stright[1] != 'wall':
                        counter+=1
                # if we are already in the goal cell
                if a < size:
                    if self.clarification_lv >= self.goal_cl_lv:
                        est_events = [event for event in estimation.cause if "I" not in event.get_signs_names()]
                        ce_events = [event for event in self.goal_pm.cause if "I" not in event.get_signs_names()]
                        for event in est_events:
                            for ce in ce_events:
                                if event.resonate('image', ce):
                                    counter += 1
                                    break
                    elif self.clarification_lv < self.goal_cl_lv:
                        if stright[1] is None:
                        # choose direction closely to  goal direction
                            closely_to_stright = ['cell'+el[-2:] for el,desc in
                                              self.additions[1]['region'+stright[0].name[-2:]].items() if desc[0] == 'closely']
                            closely_to_stright.remove('cell-4')
                            for cell in closely_to_stright:
                                if 0 not in self.additions[2]["I"][iteration][cell]:
                                    break
                            else:
                                counter+=3
                            directions = []
                            for reg, tup in self.additions[1]['region-4'].items():
                                if tup[1] == self.goal_state[self.I_obj.name]['orientation']:
                                    regs_to_goal = [reg for reg, tup2 in self.additions[1][reg].items() if tup2[0] == 'closely']
                                    directions = [tup[1] for reg, tup in self.additions[1]['region-4'].items() if reg in regs_to_goal]
                                    break
                            if current_direction.name in directions:
                                counter+=2
                            if prev_act == 'rotate' and script.sign.name == 'move':
                                counter+=2
                            elif prev_act == 'rotate' and script.sign.name == 'rotate':
                                counter = 0
                    if script.sign.name != 'rotate' and script.sign.name != 'move':
                        counter+=5
                # we are in region, but not in cell.
                else:
                    if current_direction.name == self.goal_state[self.I_obj.name]['orientation']:
                        counter += 2
                    if path <= a:
                        counter += 3
                    if prev_act == 'rotate' and script.sign.name == 'rotate':
                        counter = 0
                    if script.sign.name != 'rotate' and script.sign.name != 'move':
                        for event in self._applicable_events(estimation):
                            for ce in self._applicable_events(self.goal_pm):
                                if event.resonate('image', ce):
                                    counter += 1
                                    break
                    if 'subplan' in script.sign.name:
                        counter+=5
        return counter , path

    def _check_activity_backward_spat(self, cm, active_cm, agent, iteration):
        # TODO to precedents
        result = True
        side_active = ''
        side_cm = ''
        orient_active_cm = active_cm.get_iner(self.world_model['orientation'], 'meaning')[0]

        for sign in orient_active_cm.get_signs():
            if sign != agent:
                side_active = sign.name
                break

        orient_cm = None
        or_sign = self.world_model['orientation']
        for event in cm.effect:
            if or_sign in event:
                for connector in event.coincidences:
                    if connector.out_sign == or_sign:
                        orient_cm = getattr(connector.out_sign, 'meanings')[connector.out_index]
                        break

        for sign in orient_cm.get_signs():
            if sign != agent:
                side_cm = sign.name
                break

        if side_active != side_cm:
            return False, cm

        if cm.sign.name == 'move':
            # find mirror cell
            goal_reg = [reg for reg, place in self.additions[1]['region-4'].items() if place[1] == side_cm][0]
            mirror_side = [place[1] for reg, place in self.additions[1][goal_reg].items() if reg == 'region-4'][0]
            mirror_reg = [reg for reg, place in self.additions[1]['region-4'].items() if place[1] == mirror_side][0]
            mirror_cell = 'cell-' + mirror_reg.split('-')[-1]
            # if mirror cell is free - we could go from it
            if 0 not in self.additions[2]["I"][iteration][mirror_cell]:
                result = False
            # if goal cell not in action - result = False
            goal_cell = 'cell-' + goal_reg.split('-')[-1]
            chains = cm.spread_down_activity('meaning', 4)
            last_signs = [chain[-1].sign.name for chain in chains]
            if not goal_cell in last_signs:
                result = False

        return result, cm




















