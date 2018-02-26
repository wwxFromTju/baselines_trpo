import gym

env_names = ['Ant-v1', 'Hopper-v1', 'Humanoid-v1', 'HumanoidStandup-v1', 'Swimmer-v1', 'Walker2d-v1']

for env_name in env_names:
    env = gym.make(env_name)
    ac_space = env.action_space
    print(env_name, ac_space)