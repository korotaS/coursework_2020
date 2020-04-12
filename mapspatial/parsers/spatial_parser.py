class Problem:
    def __init__(self, domain, problem_parsed, problem_file):
        """
        name: The name of the problem
        domain: The domain in which the problem has to be solved
        init: A list of facts describing the initial state
        goal: A list of facts describing the goal state
        map: A tree of facts describing map
        constraints: A list of facts describing agents' capabilities
        """
        self.name = problem_parsed['task-name']
        self.domain = domain
        self.initial_state = problem_parsed['global-start']
        self.goal_state = problem_parsed['global-finish']
        self.map = problem_parsed['map']
        try:
            self.vanished = self.map['vanished']
        except KeyError:
            self.vanished = {}
        self.task_file = problem_file
        try:
            self.constraints = problem_parsed['constraints']
        except KeyError:
            self.constraints = {}
        self.agents = self._get_agents()

    def __repr__(self):
        return ('< Problem definition: %s '
                'Domain: %s Initial State: %s Goal State : %s >' %
                (self.name, self.domain['domain-name'],
                 self.initial_state,
                 self.goal_state))

    __str__ = __repr__

    def _get_agents(self):
        agents = []
        if self.constraints:
            agents.extend(list(self.constraints.keys()))
        else:
            for el, struct in self.initial_state.items():
                if 'ag' in el and struct['orientation']:
                    agents.append(el)

        return agents