import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch

from agent import Agent
from replayBuffer import ReplayBuffer
from configReader import *
from util import Utils

class Maddpg():
    """MADDPG Agent : Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize a MADDPG Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        
        super(Maddpg, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.random_seed = random_seed
        # Instantiate Multiple  Agent
        self.agents = [ Agent(state_size,action_size, random_seed) 
                       for i in range(num_agents) ]
        

        self.device = Utils.getDevice()
    
    def reset(self):
        """Reset all the agents"""
        for agent in self.agents:
            agent.reset()

    def act(self, states, noise = True):
        """Return action to perform for each agents (per policy)"""    
        actions = []
        for index, agent in enumerate(self.agents):
            actions.append(agent.act(states[index], add_noise = noise))
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        """ # Save experience in replay memory, and use random sample from buffer to learn"""
        for i,agent in enumerate(self.agents):
            agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def save(self):
        """Save checkpoints for all Agents"""
        for idx, agent in enumerate(self.agents):
            actor_local_filename = 'model_dir/checkpoint_actor_' + str(idx) + '.pth'
            critic_local_filename = 'model_dir/checkpoint_critic_' + str(idx) + '.pth'                    
            torch.save(agent.actor_local.state_dict(), actor_local_filename) 
            torch.save(agent.critic_local.state_dict(), critic_local_filename) 

    def load(self):
        for idx, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('model_dir/checkpoint_actor_' + str(idx)+'.pth',map_location= 'cpu'))
            agent.critic_local.load_state_dict(torch.load('model_dir/checkpoint_critic_' + str(idx)+'.pth',map_location= 'cpu'))

