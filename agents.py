from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Define Ensemble Agent
class SimpleAvgEnsembleAgent:
    def __init__(self, ppo_model, a2c_model, ddpg_model, sac_model, td3_model):
        self.ppo_model = ppo_model
        self.a2c_model = a2c_model
        self.ddpg_model = ddpg_model
        self.sac_model = sac_model
        self.td3_model = td3_model
    
    def predict(self, obs):
        ppo_action, _ = self.ppo_model.predict(obs)
        a2c_action, _ = self.a2c_model.predict(obs)
        ddpg_action, _ = self.ddpg_model.predict(obs)
        sac_action, _ = self.sac_model.predict(obs)
        td3_action, _ = self.td3_model.predict(obs)
        
        # Average the actions
        ensemble_action = np.mean([ppo_action, a2c_action, ddpg_action, sac_action, td3_action], axis=0)
        return ensemble_action

class WeightedAvgEnsembleAgent:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    
    def predict(self, obs):
        # final action is a weighted sum of actions according to calculated weights
        action = np.zeros((1, 30))
        for agent in list(self.models.keys()):
            agent_action, _ = self.models[agent].predict(obs)
            action += (self.weights[agent] * agent_action)
        
        # Clamp the values in the action array to the range [-1, 1]
        action = np.clip(action, -1, 1) 

        return action

# Consider setting the seed for reproducability

# Define PPO Agent
class PPOAgent:
    def __init__(self, env=None, total_timesteps=None, load=False):
        if not load:
            self.model = PPO("MlpPolicy", env, verbose=1, seed=1)
            self.model.learn(total_timesteps=total_timesteps)
            self.model.save("ppo")
        else:
            self.model = PPO.load("ppo")
    
    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action
    
# Define A2C Agent
class A2CAgent:
    def __init__(self, env=None, total_timesteps=None, load=False):
        if not load:
            self.model = A2C("MlpPolicy", env, verbose=1, seed=1)
            self.model.learn(total_timesteps=total_timesteps)
            self.model.save("a2c")
        else:
            self.model = A2C.load("a2c")
    
    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action
    
# Define DDPG Agent
class DDPGAgent:
    def __init__(self, env=None, total_timesteps=None, load=False):
        if not load:
            self.model = DDPG("MlpPolicy", env, verbose=1, seed=1)
            self.model.learn(total_timesteps=total_timesteps)
            self.model.save("ddpg")
        else:
            self.model = DDPG.load("ddpg")
    
    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action
     
# Define SAC Agent
class SACAgent:
    def __init__(self, env=None, total_timesteps=None, load=False):
        if not load:
            self.model = SAC("MlpPolicy", env, verbose=1, seed=1)
            self.model.learn(total_timesteps=total_timesteps)
            self.model.save("sac")
        else:
            self.model = SAC.load("sac")
    
    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action
    
# Define TD3 Agent
class TD3Agent:
    def __init__(self, env=None, total_timesteps=None, load=False):
        if not load:
            self.model = TD3("MlpPolicy", env, verbose=1, seed=1)
            self.model.learn(total_timesteps=total_timesteps)
            self.model.save("td3")
        else:
            self.model = TD3.load("td3")
    
    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

class MetaRfAgent:
    def __init__(self, ppo_model, a2c_model, ddpg_model, sac_model, td3_model, meta_model):
        self.ppo_model = ppo_model
        self.a2c_model = a2c_model
        self.ddpg_model = ddpg_model
        self.sac_model = sac_model
        self.td3_model = td3_model
        self.meta_model = meta_model
    
    def predict(self, obs):
        ppo_action, _ = self.ppo_model.predict(obs)
        a2c_action, _ = self.a2c_model.predict(obs)
        ddpg_action, _ = self.ddpg_model.predict(obs)
        sac_action, _ = self.sac_model.predict(obs)
        td3_action, _ = self.td3_model.predict(obs)

        combined_actions = np.concatenate((ppo_action, a2c_action, ddpg_action, sac_action, td3_action))
        combined_actions = combined_actions.reshape(1, -1)
        print(combined_actions.shape)
        
        # Predict best action
        predicted_actions = self.meta_model.predict(combined_actions)
        print(predicted_actions)

        return predicted_actions

