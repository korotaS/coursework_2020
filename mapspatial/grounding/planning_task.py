import sys

from mapcore.swm.src.components.semnet import Sign
from mapcore.swm.src.components.sign_task import *
from mapspatial.grounding import utils as ut

DEFAULT_FILE_PREFIX = 'wmodel_'
DEFAULT_FILE_SUFFIX = '.swm'

SIT_COUNTER = 0
SUBPLAN_COUNTER = 0
SIT_PREFIX = 'situation_'
PLAN_PREFIX = 'action_'
MAP_PREFIX = 'map_'
SUBPLAN_PREFIX = 'subplan_'
sys.setrecursionlimit(3500)


def load_signs(agent, file_name=None, load_all=False):
    signs = None
    if not file_name:
        file_name = []
        for f in os.listdir(os.getcwd()):
            if f.startswith(DEFAULT_FILE_PREFIX) and 'spatial' in f:
                if f.split(".")[0].endswith(agent) or f.split(".")[0].endswith('agent'):
                    file_name.append(f)
    else:
        file_name = [file_name]
    if file_name:
        if load_all:
            pass
        else:
            newest = 0
            file_load = ''
            for file in file_name:
                file_signature = int(''.join([i if i.isdigit() else '' for i in file]))
                if file_signature > newest:
                    newest = file_signature
                    file_load = file
            signs = pickle.load(open(file_load, 'rb'))
    else:
        logging.debug('Файл опыта для агента %s не найден' % agent)
        return None
    return signs


class SpTask:

    def __init__(self, name, signs, agent_state,
                additions, initial_state, goal_state, static_map, plagent):
        self.name = name
        self.signs = signs
        self.start_situation = agent_state['I']['start-sit']
        self.goal_situation = agent_state['I']['goal-sit']
        self.goal_map = agent_state['I']['goal-map']
        self.start_map = agent_state['I']['start-map']
        self.additions = additions
        self.initial_state = initial_state
        self.static = static_map
        self.goal_state = goal_state
        self.actions = []
        self.goal_cl_lv = additions[0][0][plagent]['cl_lv_goal']
        # create during planning - if None - from additions, or from prev plan
        self.init_cl_lv = None
        # push iteration to next planning action in multiagent version
        self.iteration = 0
        # for inheritance in mapsearch
        self.subtasks = {}

    def __str__(self):
        s = 'Task {0}\n  Signs:  {1}\n  Start:  {2}\n  Goal: {3}\n'
        return s.format(self.name, '\n'.join(map(repr, self.signs)),
                        self.start_situation, self.goal_situation)

    def __repr__(self):
        return '<Task {0}, signs: {1}>'.format(self.name, len(self.signs))

    def save_signs(self, plans):
        """
        Cleaning SWM and saving experience
        :param plan:
        :return:
        """
        # with open('solution.txt', 'w', encoding='utf-8') as sol:
        #     data = [(el[1], el[3].name, el[4][0]['x'], el[4][0]['y'], el[4][1]) for el in plan]
        #     for element in data:
        #         sol.write(str(element)+'\n')

        logging.info('Начинаю подготовку к сохранению плана...')

        def __is_role(pm, agents):
            chains = pm.spread_down_activity('meaning', 6)
            for chain in chains:
                if chain[-1].sign not in agents:
                    if len(chain[-1].sign.significances[1].cause) != 0:
                        break
            else:
                return False
            return True


        def create_situation(description, agent, whose = 'They', size_prev = None):
            region_map, cell_map_pddl, cell_location, near_loc, cell_coords, size, cl_lv = ut.signs_markup(description,
                                                                                                        self.additions[
                                                                                                            3],
                                                                                                        agent.name, size=size_prev)
            if cl_lv < description['cl_lv'] and not size_prev:
                #it can be only iff actions's cl_lv more than after markup
                agcx = description['objects'][agent.name]['x']
                agcy = description['objects'][agent.name]['y']
                while cl_lv != description['cl_lv']:
                    size_x = (size[2] - size[0]) // 6
                    size_y = (size[3] - size [1]) // 6
                    size = [agcx - size_x, agcy - size_y, agcx + size_x, agcy + size_y]
                    cl_lv+=1
                region_map, cell_map_pddl, cell_location, near_loc, cell_coords, size, cl_lv = ut.signs_markup(
                    description,
                    self.additions[
                        3],
                    agent.name, size=size)
            agent_state_action = ut.state_prediction(agent, description, self.signs)
            conditions_new = description['conditions']
            global SIT_COUNTER
            SIT_COUNTER+=1
            act_sit = ut.define_situation('situation_'+whose+'_' + str(SIT_COUNTER), cell_map_pddl, conditions_new,
                                                agent_state_action, self.signs)
            act_map = ut.define_map('map_'+whose+'_' + str(SIT_COUNTER), region_map, cell_location, near_loc,
                                    self.additions[1],
                                    self.signs)
            ut.state_fixation(act_sit, cell_coords, self.signs, 'cell')

            return act_sit, act_map, size

        def create_subplan(subplan, agent, signs, plan_sit, plan_map, start = False):
            elem_acts_im = []
            if agent == 'I':
                ag_sign = I_obj[0]
                whose = agent
            else:
                ag_sign = signs[agent]
                whose = 'They'
            act_st = subplan[0]
            if isinstance(act_st[-1], dict):
                descr1 = act_st[-1]
            else:
                descr1 = act_st[-1][0]
            # Remake situations coze they could be synthesized by other agent
            act_sit_start, act_map_start, size = create_situation(descr1, ag_sign, whose)
            if agent == 'I':
                I_sign = signs['I']
                I_im = I_sign.add_image()
                act_sit_start.replace('image', ag_sign, I_im)
            plan_map.append(act_map_start.sign)
            act_fn = subplan[-1]
            if isinstance(act_fn[-1], dict):
                logging.debug('cant find goal sit for subplan')
                descr2 = act_fn[-1]
            else:
                descr2 = act_fn[-1][1]
            # Change goal sit size
            ag_x = descr1['objects'][ag_sign.name]['x']
            ag_y = descr1['objects'][ag_sign.name]['y']
            for el in subplan:
                if el[1] == 'Clarify':
                    new_size = (size[2] - size[0]) // 6
                    size = [ag_x-new_size, ag_y-new_size, ag_x+new_size, ag_y+new_size]
                elif el[1] == 'Abstract':
                    new_size = size[2] - size[0]
                    size =  [size[0]-new_size, size[1] - new_size, size[2]+new_size, size[3]+new_size]
            act_sit_finish, act_map_finish, _ = create_situation(descr2, ag_sign, whose, size)
            plan_map.append(act_map_finish.sign)
            if agent == 'I':
                I_sign = signs['I']
                I_im = I_sign.add_image()
                act_sit_finish.replace('image', ag_sign, I_im)
            active_sit_start_mean = act_sit_start.copy('image', 'meaning')
            plan_sit.append(act_sit_start.sign)
            active_sit_finish_mean = act_sit_finish.copy('image', 'meaning')
            plan_sit.append(act_sit_finish.sign)
            global SUBPLAN_COUNTER
            SUBPLAN_COUNTER+=1
            if agent != 'I':
                plan_sign = Sign(SUBPLAN_PREFIX+'they_'+agent+'_'+str(SUBPLAN_COUNTER))
                plan_mean = plan_sign.add_meaning()
                connector = plan_mean.add_feature(active_sit_start_mean)
                active_sit_start_mean.sign.add_out_meaning(connector)
                conn = plan_mean.add_feature(active_sit_finish_mean, effect=True)
                active_sit_finish_mean.sign.add_out_meaning(conn)
                signs[plan_sign.name] = plan_sign
            else:
                plan_sign = Sign(SUBPLAN_PREFIX+'I_'+str(SUBPLAN_COUNTER))
                plan_mean = plan_sign.add_meaning()
                connector = plan_mean.add_feature(active_sit_start_mean)
                active_sit_start_mean.sign.add_out_meaning(connector)
                conn = plan_mean.add_feature(active_sit_finish_mean, effect=True)
                active_sit_finish_mean.sign.add_out_meaning(conn)
                plan_image = plan_sign.add_image()
                for act in subplan:
                    act_sign = self.signs[act[1]]
                    im = act_sign.add_image()
                    connector = plan_image.add_feature(im)
                    act_sign.add_out_image(connector)
                    elem_acts_im.append(im)
                signs[plan_sign.name] = plan_sign
            ## changing start and finish
            if start == True:
                self.start_situation = act_sit_start
                self.start_map = act_map_start
            self.goal_situation = act_sit_finish
            self.goal_map = act_map_finish
            return [act_sit_start, plan_sign.name, plan_mean, act_st[3], (None, None),
                         (act_map_start, None),
                         (descr1, descr2)], signs, plan_sit, plan_map, elem_acts_im

        I_obj = [con.in_sign for con in self.signs["I"].out_significances if con.out_sign.name == "I"]
        if plans:
            They_signs = [con.in_sign for con in self.signs["They"].out_significances]
            agents = [self.signs["I"]]
            agents.extend(They_signs)

            logging.debug('\tCleaning SWM...')

            pms_acts = []
            goal_plan = []
            subplans = []
            plan_sit = []
            plan_map = []
            elem_acts_ims = []
            """
            First - save all detailed implementations of classic actions as subplans. 
            Find in this implementations subsubplans as Clarify --- Abstract (thin moves) and save them too.
            """
            for action in plans:
                for glob_act, plan in action.items():
                    logging.info("Сохраняю подплан для действия {0} агента {1}".format(glob_act[0], glob_act[1]))
                    if plans.index(action) == 0:
                        subplan, self.signs, plan_sit, plan_map, elem_acts_im = create_subplan(plan, glob_act[1], self.signs, plan_sit, plan_map, start = True)
                    else:
                        subplan, self.signs, plan_sit, plan_map, elem_acts_im = create_subplan(plan, glob_act[1],
                                                                                               self.signs, plan_sit,
                                                                                               plan_map)
                    subplans.append(subplan)
                    if glob_act[1] == 'I':
                        # images of elementary actions that are used in subplan construction
                        elem_acts_ims.extend(elem_acts_im)
                        goal_plan.extend(plan)
                        plan_sit.extend([pm[0].sign for pm in plan])
                        plan_map.extend([pm[5][0].sign for pm in plan])
                        pms_act = [pm[2] for pm in plan]
                        pms_acts.extend(pms_act)
                        # Add start and finish situations of Clarification and Abstraction or subplan to plan sits
                        for element in plan:
                            if element[2].sign.name == 'Clarify' or element[2].sign.name == 'Abstract':
                                if len(element[2].effect) == 1:
                                    start = list(element[2].cause[0].get_signs())[0]
                                    finish = list(element[2].effect[0].get_signs())[0]
                                else:
                                    mean = None
                                    for ind, m in element[2].sign.meanings.items():
                                        if len(m.effect) == 1:
                                            mean = m
                                            break
                                    else:
                                        logging.debug('Can not find matrice with link to situation from images of sign {}'.format(
                                            element[1].sign.name))
                                    start = list(mean.cause[0].get_signs())[0]
                                    finish = list(mean.effect[0].get_signs())[0]
                                plan_sit.append(start)
                                plan_sit.append(finish)
                            elif 'subplan_' in element[2].sign.name:
                                start = list(element[2].cause[0].get_signs())[0]
                                finish = list(element[2].effect[0].get_signs())[0]
                                plan_sit.append(start)
                                plan_sit.append(finish)
                                mapName = 'map_they_'+finish.name.split('_')[-1]
                                plan_map.append(self.signs[mapName])
                        global SUBPLAN_COUNTER
                        # If there are Clarify and Abstract in subplan - save
                        logging.info('\tПоиск уточненных частей плана %s агента... ' % I_obj[0].name)
                        if glob_act[1] == 'I':
                            pms_plan = [act[2].sign.name + ':' + str(plan.index(act)) for act in plan]
                            str_plan = ''.join(el + ' ' for el in pms_plan)
                            if 'Abstract' in str_plan and 'Clarify' in str_plan:
                                # Save plan between abstr and clarify acts iff the refinement level is identical
                                for st, end, _ in ut.tree_refinement(str_plan, opendelim='Clarify', closedelim='Abstract'):
                                    if end - st + 1 != len(pms_act):
                                        start_sign = list(pms_act[st].cause[0].get_signs())[0]
                                        if not start_sign.meanings:
                                            start_mean = start_sign.images[1].copy('image', 'meaning')
                                        else:
                                            start_mean = start_sign.meanings[1]
                                        finish_sign = list(pms_act[end].effect[0].get_signs())[0]
                                        if not finish_sign.meanings:
                                            finish_mean = finish_sign.images[1].copy('image', 'meaning')
                                        else:
                                            finish_mean = finish_sign.meanings[1]
                                        SUBPLAN_COUNTER += 1
                                        plan_sign, _, _ = self.save_plan(start_mean, finish_mean, plan[st:end + 1],
                                                                         'subplan_' + str(SUBPLAN_COUNTER))

            """
            Clear model from non plan actions and situations
            """
            for name, s in self.signs.copy().items():
                signif=list(s.significances.items())
                if name.startswith(SIT_PREFIX) or name.startswith(MAP_PREFIX):
                    for index, pm in s.meanings.copy().items():
                        if s not in plan_sit and s not in plan_map:
                            s.remove_meaning(pm) # delete all meanings of situations that are not in plan
                        elif index > 1:
                            s.remove_meaning(pm) # delete double meaning from plan situations
                    if s in plan_sit or s in plan_map: # only 1 mean and 1 image per plan sit
                        for index, im in s.images.copy().items():
                            if index > 2:
                                s.remove_view(im)
                    else:
                        for index, im in s.images.copy().items():
                            if index != 2:
                                try:
                                    s.remove_image(im) # remove other
                                except AttributeError:
                                    s.remove_view(im) # remove view
                            else:
                                s.remove_view(im) # remove view
                        self.signs.pop(name) # delete this situation

                elif len(signif):
                    if len(signif[0][1].cause) and len(signif[0][1].effect): #delete action's meanings that are not in plan
                        if not name.startswith(SUBPLAN_PREFIX):
                            for index, pm in s.meanings.copy().items():
                                if __is_role(pm, agents):  # delete only fully signed actions
                                    continue
                                else:
                                    if pm not in pms_acts:
                                        s.remove_meaning(pm)
                            for index, im in s.images.copy().items():
                                if im not in elem_acts_ims:
                                    s.remove_image(im) # delete all action's images
            # Clean situations
            global_sit = self.signs['situation']
            for index, im in global_sit.images.copy().items():
                pm_signs = im.get_signs()
                for sign in pm_signs:
                    if sign not in plan_sit:
                        global_sit.images.pop(index)

            logging.info('\tСохраняю пространственный прецедент...')

            self.save_plan(self.start_situation.sign.meanings[1], self.goal_situation.sign.meanings[1], goal_plan, '',
                            subplans)

        else:
            logging.info('План не был синтезирован. Измените условия поиска.')
            for name, sign in self.signs.copy().items():
                if name.startswith(SIT_PREFIX) or name.startswith(MAP_PREFIX):
                    self.signs.pop(name)
                else:
                    sign.meanings = {}
                    sign.out_meanings = []
                    sign.images = {}
                    sign.out_images = []
        if I_obj:
            I_obj = "_"+I_obj[0].name
        else:
            I_obj = 'I'
        file_name = DEFAULT_FILE_PREFIX + datetime.datetime.now().strftime('%m_%d_%H_%M')  + '_spatial_' + I_obj +DEFAULT_FILE_SUFFIX
        logging.info('Файл пространственного опыта: {0}'.format(file_name))
        logging.debug('\tСохраняю ЗКМ агента...')
        pickle.dump(self.signs, open(file_name, 'wb'))
        logging.info('\tСохранение выполнено.')
        return file_name

    def save_plan(self, start, finish, actions, plan_name, subplans = None):
        # Creating plan action for further use
        if not plan_name:
            plan_name = 'plan_'+ self.name
        if not start.sign.meanings:
            scm = start.copy('image', 'meaning')
            start.sign.add_meaning(scm)
        if not finish.sign.meanings:
            fcm = finish.copy('image', 'meaning')
            finish.sign.add_meaning(fcm)
        plan_sign = Sign(plan_name)
        plan_mean = plan_sign.add_meaning()
        connector = plan_mean.add_feature(start.sign.meanings[1])
        start.sign.add_out_meaning(connector)
        conn = plan_mean.add_feature(finish.sign.meanings[1], effect=True)
        finish.sign.add_out_meaning(conn)
        self.signs[plan_sign.name] = plan_sign

        # Adding Sequence of actions to plan image
        plan_image = plan_sign.add_image()
        iter = -1
        if not subplans:
            for act in actions:
                im = act[2].sign.add_image()
                connector = plan_image.add_feature(im)
                act[2].sign.add_out_image(connector)  # add connector to plan_sign threw images to out_image
        else:
            # ots = {subplans.index(it): range(it[1], it[2]+1) for it in subplans}
            # for act in actions:
            #     iter += 1
            #     flag = False
            #     for index, value in ots.items():
            #         if iter == value[0]:
            #             if subplans[index][0].images:
            #                 im = subplans[index][0].images[1]
            #             else:
            #                 im  = subplans[index][0].add_image()
            #             connector = plan_image.add_feature(im)
            #             subplans[index][0].add_out_image(connector)
            #             flag = True
            #             break
            #         elif iter in value:
            #             flag = True
            #             break
            #     if flag:
            #         continue
            #     else:
            #         im = act[2].sign.add_image()
            #         connector = plan_image.add_feature(im)
            #         act[2].sign.add_out_image(connector)
            for sub in subplans:
                if sub[2].sign.images:
                    connector = plan_image.add_feature(sub[2].sign.images[1])
                else:
                    they_im = sub[2].sign.add_image()
                    connector = plan_image.add_feature(they_im)
                sub[2].sign.add_out_image(connector)  # add connector to plan_sign threw images to out_image
        # Adding scenario vs partly concrete actions to the plan sign
        scenario = self.scenario_builder(start, finish, actions)
        plan_signif = plan_sign.add_significance()
        for act in scenario:
            connector = plan_signif.add_feature(act)
            act.sign.add_out_significance(connector)

        return [plan_sign, start.sign, finish.sign]

    def scenario_builder(self, start, goal, actions):
        scenario = []
        for act in actions:
            if act[2].sign.significances:
                scenario.append(act[2].sign.significances[1])
            else:
                signif = act[2].sign.add_significance()
                scenario.append(signif)

        return scenario

