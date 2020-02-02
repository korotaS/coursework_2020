from gym.envs.registration import register


register(
    id='GridWorld-v1',
    entry_point='envs.gridworld.envs.GridWorld:GridWorld'
)
