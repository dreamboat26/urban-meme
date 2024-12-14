from agents.agentDDPG import agentDDPG
from task import Task
import numpy as np

episodes = 20000
init_pose = np.array([0., 0., 0., 0., 0., 0.])
target_pos = np.array([0., 10., 0.])
task = Task(init_pose, target_pos)

agent = agentDDPG(task)

for i in range(episodes):
    state = agent.reset_episode()
    eps_reward = []
    total_reward = 0
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done == True:
            eps_reward.append(total_reward)
            print("Episode: {}/{}".format(i+1, episodes), "Score: {}".format(total_reward))