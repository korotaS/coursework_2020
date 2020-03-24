class Agent:
    """
    Agent 

    """

    def act(self, state):
        raise NotImplementedError()

    def update(self, state, action, reward, next_state):
        raise NotImplementedError()
