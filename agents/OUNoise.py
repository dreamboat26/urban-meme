import copy
import numpy as np

class OUNoise():
    def __init__(self, size, mu, sigma, theta):
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.reset()
        
    def reset(self):
        self.state = copy.copy(self.mu)
    
    def sample(self):
        x = self.state
        dx = self.theta*(self.mu - x) + self.sigma*np.random.randn(len(x))
        
        self.state = x + dx
        return self.state