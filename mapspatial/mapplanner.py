import logging
import os
import json
from mapcore.planning.mapplanner import MapPlanner as MPcore
from mapspatial.agent.planning_agent import Manager
from mapspatial.parsers.spatial_parser import Problem

SOLUTION_FILE_SUFFIX = '.soln'

import platform

if platform.system() != 'Windows':
    delim = '/'
else:
    delim = '\\'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process-main")

class MapPlanner(MPcore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subsearch = kwargs['Settings']['subsearch']

    def find_domain(self, domain, path, number):
        """
        Domain search function
        :param path: path to current task
        :param number: task number
        :return:
        """
        if 'spatial' in self.TaskType:
            ext = '.json'
            path += 'task' + number + delim
        elif self.TaskType == 'htn':
            ext = '.hddl'
        else:
            ext = '.pddl'
        task = 'task' + number + ext
        domain = 'domain' + ext
        if not domain in os.listdir(path):
            domain2 = self.search_upper(path, domain)
            if not domain2:
                raise Exception('domain not found!')
            else:
                domain = domain2
        else:
            domain = path + domain
        if not task in os.listdir(path):
            raise Exception('task not found!')
        else:
            problem = path + task

        return domain, problem

    def _parse_spatial(self):
        """
        spatial Parser
        :param domain_file:
        :param problem_file:
        :return:
        """
        logging.debug('Распознаю проблему {0}'.format(self.problem))
        with open(self.problem) as data_file1:
            problem_parsed = json.load(data_file1)
        logging.debug('Распознаю домен {0}'.format(self.domain))
        with open(self.domain) as data_file2:
            signs_structure = json.load(data_file2)

        logging.debug('{0} найдено объектов'.format(len(problem_parsed['global-start']['objects'])))
        logging.debug('{0} найдено предикатов'.format(len(signs_structure['predicates'])))
        logging.debug('{0} найдено действий'.format(len(signs_structure['actions'])))
        logging.info('Карта содержит {0} неперемещаемых препятствий'.format(len(problem_parsed['map']['wall'])))
        logging.info('Размер карты {0}:{1}'.format(problem_parsed['map']['map-size'][0], problem_parsed['map']['map-size'][1]))
        problem = Problem(signs_structure, problem_parsed, self.problem)
        return problem


    def search(self):
        """
        spatial - json-pddl- based plan search
        :return: the final solution
        """
        problem = self._parse_spatial()
        logger.info('Пространственная проблема получена и распознана')
        manager = Manager(problem, self.agpath, TaskType=self.TaskType, backward=self.backward, subsearch = self.subsearch)
        solution = manager.manage_agents()
        return solution

