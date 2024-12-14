from collections import namedtuple, deque
import random

class memoryBuffer():
    def __init__(self, memorySize, batchSize):
        self.memory = deque(maxlen=memorySize)
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        return random.sample(self.memory, k=self.batchSize)
    
    def len(self):
        return len(self.memory)