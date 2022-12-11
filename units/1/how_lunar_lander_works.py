import gym
from stable_baselines3.common.env_util import make_vec_env

ENVIRONMENT_NAME = "LunarLander-v2"

env = gym.make("LunarLander-v2")
env.reset()

print("____SPACE____\n")
print("Observation space shape", env.observation_space.shape)
print("Sample observatino", env.observation_space.sample()) # get a random observation

print("\n____ACTION SPACE____\n")
print("Action space shape", env.action_space.n)
print("Action space sample", env.action_space.sample()) # take a random action

# define a vectorized environment
env = make_vec_env('LunarLander-v2', n_envs=16)
