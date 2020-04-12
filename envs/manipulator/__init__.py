from gym.envs.registration import register


register(
    id='Manipulator-v1',
    entry_point='envs.manipulator.envs.Manipulator:Manipulator'
)
