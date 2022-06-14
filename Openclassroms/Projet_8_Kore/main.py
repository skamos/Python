# All this syspath wranglig is needed to make sure that the agent runs on the target environment and can load both the external dependencies
# and the saved model. Dear kaggle, if possible, please make this easier!
import os
import sys
KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_AGENT_PATH):
    # We're in the kaggle target system
    sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, 'lib'))
    agent_path = os.path.join(KAGGLE_AGENT_PATH, 'baseline_agent')
else:
    # We're somewhere else
    sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))
    agent_path = 'baseline_agent'

# Now for the actual agent
from stable_baselines3 import PPO
from environment import KoreGymEnv

model = PPO.load(agent_path)
kore_env = KoreGymEnv()

def agent(obs, config):
    kore_env.raw_obs = obs
    state = kore_env.obs_as_gym_state
    action, _ = model.predict(state)
    return kore_env.gym_to_kore_action(action)
