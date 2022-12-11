import gym

ENVIRONMENT_NAME = "LunarLander-v2"

# create a new environment
env = gym.make(ENVIRONMENT_NAME)

# reset the environment
observation = env.reset()

# step through 20 actions
for _ in range(20):

    # take a random action
    action = env.action_space.sample()
    print("Action taken:", action)

    # execute action in environment
    observation, reward, done, info = env.step(action)

    # end if terminal step
    if done:
        # reset environment
        print('environment is reset')
        observation = env.reset()

