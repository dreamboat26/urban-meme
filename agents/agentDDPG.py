from agents.actor import actor
from agents.critic import critic
from agents.OUNoise import OUNoise
from agents.memoryBuffer import memoryBuffer
import numpy as np

class agentDDPG():
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.rotor_high = task.action_high
        self.rotor_low = task.action_low
        
        # We will update local agent continuously and intermittantly copy the weights to target agent
        self.actor_local = actor(self.state_size, self.action_size, h1=64, h2=32, lr=0.001, r_h=self.rotor_high, r_l=self.rotor_low)
        self.actor_target = actor(self.state_size, self.action_size, h1=64, h2=32, lr=0.001, r_h=self.rotor_high, r_l=self.rotor_low)
        
        self.critic_local = critic(self.state_size, self.action_size, h1=32, h2=24, lr=0.001)
        self.critic_target = critic(self.state_size, self.action_size, h1=32, h2=24, lr=0.001)
        
        # Make the weights of both local and target agent same
        self.actor_target.actorModel.set_weights(self.actor_local.actorModel.get_weights())
        self.critic_target.criticModel.set_weights(self.critic_local.criticModel.get_weights())
        
        self.mu = 0
        self.sigma = 0.15
        self.theta = 0.2
        self.OUNoise = OUNoise(self.action_size, self.mu, self.sigma, self.theta)
        
        self.bufferSize = 100000
        self.batch_size = 64
        self.memory = memoryBuffer(self.bufferSize, self.batch_size)
        
        self.gamma = 0.99
        self.tau = 0.01
        
    def reset_episode(self):
        self.OUNoise.reset()
        state = self.task.reset()
        self.last_state = state
        return state
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        if self.memory.len() > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            
        self.last_state = next_state

    def learn(self, experience):
        states = np.vstack([e.state for e in experience if e is not None])
        actions = np.array([e.action for e in experience if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experience if e is not None]).astype(np.float32).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experience if e is not None])
        done = np.array([e.done for e in experience if e is not None]).astype(np.uint8).reshape(-1, 1)
        
        
        # Train actor agent based on the action_gradient received from critic
        # Train critic agent based on TD error
        actions_next = self.actor_local.actorModel.predict_on_batch(next_states)
        Q_targets_next = self.critic_local.criticModel.predict_on_batch([next_states, actions_next])
        
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - done)
        self.critic_local.criticModel.train_on_batch(x=[states, actions], y=Q_targets)
        
        action_gradients = self.critic_local.get_action_gradients(inputs=[states, actions, 0])
        action_gradients = np.reshape(action_gradients, (-1, self.action_size))
        
        self.actor_local.train_actor(inputs=[states, action_gradients, 1])
        
        self.soft_update(self.actor_local.actorModel, self.actor_target.actorModel)
        self.soft_update(self.critic_local.criticModel, self.critic_target.criticModel)
        
    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        
        assert len(local_weights) == len(target_weights)
        
        new_weights = local_weights * self.tau + target_weights * (1 - self.tau)
        target_model.set_weights(new_weights)
        
    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.actorModel.predict(state)[0]
        return list(action + self.OUNoise.sample())