# modules & functions
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# params
GYM_ENVIRONMENT = "LunarLander-v2"
MODEL_NAME = "lunar_lander_v2"
TRAIN_STEPS = int(1e6)
SAVE = False

# create environment
# env = gym.make(GYM_ENVIRONMENT)
env = make_vec_env(GYM_ENVIRONMENT, n_envs=16)

# train environment using proximal policy
model = PPO(MODEL_NAME, env, verbose=-1)
model.learn(total_timesteps=TRAIN_STEPS)

# save model to file
if SAVE:
    model.save(MODEL_NAME)

# evaluate model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=True)
print(f"mean_reward {mean_reward:.2f}+/-{std_reward:.2f}")
