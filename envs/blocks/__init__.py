from gym.envs.registration import register


register(
    id='BlocksWorld-v1',
    entry_point='envs.blocks.envs.BlocksWorld:BlocksWorld'
)
