import logging
import sys
from copy import deepcopy
from mapcore.swm.src.components.semnet import Sign
from mapspatial.grounding.planning_task import SpTask

from mapspatial.grounding import utils as ut

signs = {}

def spatial_ground(problem, plagent, agents, exp_signs=None, backward = False):
    global signs
    initial_state = problem.initial_state
    initial_state.update(problem.map)
    goal_state = problem.goal_state
    goal_state.update(problem.map)

    # Prepare states to plagent
    init_state = {key: value for key, value in deepcopy(initial_state).items() if key != plagent}
    init_state['I'] = initial_state[plagent]
    init_state['objects']['I'] = init_state['objects'].pop(plagent)
    go_state = {key: value for key, value in deepcopy(goal_state).items() if key != plagent}
    go_state['I'] = goal_state[plagent]
    go_state['objects']['I'] = go_state['objects'].pop(plagent)
    agents.remove(plagent)
    agents.append('I')

    obj_signifs = {}
    obj_means = {}
    # Create agents and communications
    agent_type = None

    types = problem.domain['types']
    roles = problem.domain['roles']

    if exp_signs:
        signs, types = ut._clear_model(exp_signs, signs, types)
        signs, roles = ut._clear_model(exp_signs, signs, roles)
        I_sign = exp_signs['I']
        signs['I'] = I_sign
        They_sign = exp_signs['They']
        signs['They'] = They_sign
        Clarify = exp_signs['Clarify']
        signs['Clarify'] = Clarify
        Abstract = exp_signs['Abstract']
        signs['Abstract'] = Abstract
        signs['situation'] = exp_signs['situation']
    else:
        I_sign = Sign("I")
        They_sign = Sign("They")
        obj_means[I_sign] = I_sign.add_meaning()
        obj_signifs[I_sign] = I_sign.add_significance()
        signs[I_sign.name] = I_sign
        obj_means[They_sign] = They_sign.add_meaning()
        obj_signifs[They_sign] = They_sign.add_significance()
        signs[They_sign.name] = They_sign
        Clarify = Sign('Clarify')
        signs[Clarify.name] = Clarify
        Abstract = Sign('Abstract')
        signs[Abstract.name] = Abstract
        signs['situation'] = Sign('situation')

    # ground types
    for type_name, smaller in types.items():
        if not type_name in signs and not exp_signs:
            type_sign = Sign(type_name)
            signs[type_name] = type_sign
        else:
            if exp_signs:
                type_sign = exp_signs[type_name]
            else:
                type_sign = signs[type_name]
        if smaller:
            for obj in smaller:
                if not obj in signs:
                    obj_sign = Sign(obj)
                    signs[obj] = obj_sign
                else:
                    obj_sign = signs[obj]

                obj_signif = obj_sign.add_significance()
                obj_signifs[obj_sign] = obj_signif
                tp_signif = type_sign.add_significance()
                connector = tp_signif.add_feature(obj_signif, zero_out=True)
                obj_sign.add_out_significance(connector)
                # Assign I to agent
                if obj_sign.name == plagent:
                    connector = obj_signif.add_feature(obj_signifs[I_sign], zero_out=True)
                    I_sign.add_out_significance(connector)
                    agent_type = type_name
        else:
            obj_signifs[type_sign] = type_sign.add_significance()

    # Assign other agents
    others = {signs[ag] for ag in agents if ag != 'I'}

    for subagent in others:
        if not They_sign in subagent.significances[1].get_signs():
            signif = obj_signifs[They_sign]
            if signif.is_empty():
                They_signif = signif
            else:
                They_signif = They_sign.add_significance()
            connector = subagent.significances[1].add_feature(They_signif, zero_out=True)
            They_sign.add_out_significance(connector)
            obj_means[subagent] = subagent.add_meaning()
    # Signify roles
    for role_name, smaller in roles.items():
        role_sign = Sign(role_name)
        signs[role_name] = role_sign

        for object in smaller:
            obj_sign = signs[object]
            obj_signif = obj_sign.significances[1]
            role_signif = role_sign.add_significance()
            connector = role_signif.add_feature(obj_signif, zero_out=True)
            obj_sign.add_out_significance(connector)
            if object == agent_type:
                agent_type = role_name
    # Find and signify walls
    if exp_signs:
        if not 'wall' in signs:
            signs['wall'] = exp_signs['wall']
    ws = signs['wall']
    views = []
    if ws.images:
        for num, im in ws.images.items():
            if len(im.cause):
                for view in im.cause[0].coincidences:
                    if view.view:
                        views.append(view.view)
    if 'wall' in problem.map:
        for wall in problem.map['wall']:
            if wall not in views:
                cimage = ws.add_image()
                cimage.add_feature(wall, effect=False, view = True)
    else:
        logging.warning('There are no walls around! Check your task!!!')
        sys.exit(1)
    # Ground predicates and actions
    if not exp_signs:
        ut._ground_predicates(problem.domain['predicates'], signs)
        ut._ground_actions(problem.domain['actions'], obj_means, problem.constraints, signs, agents, agent_type)
    else:
        #TODO check if new action or predicate variation appeared in new task
        for pred in problem.domain['predicates'].copy():
            if pred in exp_signs:
                signs[pred] = exp_signs[pred]
                problem.domain['predicates'].pop(pred)
        for act in problem.domain['actions'].copy():
            if act in exp_signs:
                signs[act] = exp_signs[act]
                problem.domain['actions'].pop(act)
        ut._ground_predicates(problem.domain['predicates'], signs)
        ut._ground_actions(problem.domain['actions'], obj_means, problem.constraints, signs, agents, agent_type)
        #Copy all experience subplans and plans
        for sname, sign in exp_signs.items():
            if sname not in signs:
                signs[sname] = sign

    # Define start situation
    maps = {}
    if backward:
        maps[0] = problem.goal_state
    else:
        maps[0] = problem.initial_state
    ms = maps[0].pop('map-size')
    walls = maps[0].pop('wall')
    static_map = {'map-size': ms, 'wall': walls}

    # Create maps and situations for planning agents
    regions_struct = ut.get_struct()
    additions = []
    additions.extend([maps, regions_struct, static_map])
    cells = {}
    agent_state = {}

    for agent in agents:
        region_map, cell_map_start, cell_location, near_loc, cell_coords, size, cl_lv_init = ut.signs_markup(init_state, static_map,
                                                                                               agent)
        agent_state_start = ut.state_prediction(signs[agent], init_state, signs)
        start_situation = ut.define_situation('*start-sit*-'+agent, cell_map_start, problem.initial_state['conditions'], agent_state_start, signs)
        start_map = ut.define_map('*start-map*-'+agent, region_map, cell_location, near_loc, regions_struct, signs)
        ut.state_fixation(start_situation, cell_coords, signs, 'cell')

        # Define goal situation
        region_map, cell_map_goal, cell_location, near_loc, cell_coords, size, cl_lv_goal = ut.signs_markup(go_state, static_map,
                                                                                               agent)
        agent_state_finish = ut.state_prediction(signs[agent], go_state, signs)
        goal_situation = ut.define_situation('*goal-sit*-'+agent, cell_map_goal, problem.goal_state['conditions'], agent_state_finish, signs)
        goal_map = ut.define_map('*goal-map*-'+agent, region_map, cell_location, near_loc, regions_struct, signs)
        ut.state_fixation(goal_situation, cell_coords, signs, 'cell')

        #fixation map
        map_size = ut.scale(ms)
        rmap = [0, 0, map_size[0], map_size[1]]
        region_location, _ = ut.locater('region-', rmap, initial_state['objects'], walls)
        ut.state_fixation(start_map, region_location, signs, 'region')
        ut.signify_connection(signs)
        if agent == 'I':
            additions[0][0][plagent]['cl_lv_init'] = cl_lv_init
            additions[0][0][plagent]['cl_lv_goal'] = cl_lv_goal
        else:
            additions[0][0][agent]['cl_lv_init'] = cl_lv_init
            additions[0][0][agent]['cl_lv_goal'] = cl_lv_goal
        if backward:
            cell_map_goal['cell-4'] = {plagent}
            cells[agent] = {0:cell_map_goal}
        else:
            cell_map_start['cell-4'] = {plagent}
            cells[agent] = {0: cell_map_start}
        agent_state[agent] = {'start-sit':start_situation.sign, 'goal-sit': goal_situation.sign, 'start-map':start_map.sign,
                              'goal-map': goal_map.sign}

    additions.insert(2, cells)

    return SpTask(problem.name, signs, agent_state, additions, problem.initial_state,
                  {key:value for key, value in problem.goal_state.items() if key not in static_map}, static_map, plagent)
