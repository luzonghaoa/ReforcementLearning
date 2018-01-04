import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np
import gym
import math
import matplotlib.pyplot as plt
from gym import wrappers

# parameter
MAX_T = 2000
MAX_EPISODE = 150
BATCH_SIZE = 32
TARGET_ITER = 100
LR = 0.01                   # learning rate
#EPSILON = 0.6              # explore rate
MIN_EPSILON = 0.05
GAMMA = 0.9
MEMORY_SIZE = 2000          # size of memory
hidden_n = 5

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_n)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(hidden_n, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        q_value = self.out(x)
        return q_value

class DQN(object):
    def __init__(self, env):
        self.env = env
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        #print(self.n_states, self.n_actions, self.n_states * 2 + 3)
        self.net = Net(self.n_states, self.n_actions)
        self.target_net = Net(self.n_states, self.n_actions)
        #self.target_net = self.net
        #print(list(self.net.parameters())[1])
        #print(list(self.target_net.parameters())[1])
        self.c = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_SIZE, self.n_states * 2 + 3))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()   # loss calculate

    def get_action(self, state, EPSILON):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        if random.random() < EPSILON:
            action = self.env.action_space.sample()
        else:
            actions_value = self.net.forward(state)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        return action

    def store_memory(self, s, a, r, d, s_):
        transition = np.hstack((s, [a, r, d], s_))
        # loop store
        index = self.memory_counter % MEMORY_SIZE
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample_memory(self):
        sample_index = np.random.choice(MEMORY_SIZE, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2]))
        b_done = b_memory[:, self.n_states + 2:self.n_states + 3]
        #print(b_done)
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_states:]))
        return b_s, b_a, b_r, b_done, b_s_

def get_explore_rate(t):
    return max(MIN_EPSILON, min(1, 1.0 - math.log10((t + 1) / 25)))

def learn(model):
    env = gym.make(model)
    env = env.unwrapped
    dqn = DQN(env)
    #EPSILON = get_explore_rate(0)
    EPSILON = 0.5
    point = []
    point_r = []
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        cnt = 0
        tmp = 0
        tmp_r = 0
        for t in range(MAX_T):
            #env.render()
            a = dqn.get_action(s, EPSILON)
            s_, r, done, info = env.step(a)

            # store memory
            dqn.store_memory(s, a, r, done, s_)

            if dqn.memory_counter > MEMORY_SIZE:
                if dqn.c % TARGET_ITER == 0:
                    dqn.target_net.load_state_dict(dqn.net.state_dict())
                dqn.c += 1
                s_j, a_j, r_j, done_j, s_j_ = dqn.sample_memory()
                q_j = dqn.net(s_j).gather(1, a_j.view(-1, 1))
                y = np.zeros(BATCH_SIZE)
                for i in range(BATCH_SIZE):
                    # calculate yj
                    if done_j[i]:
                        y[i] = r_j[i]
                    else:
                        #print(s_j_)
                        q_t_ = dqn.target_net(s_j_).max(1)[0]
                        y[i] = (r_j[i] + (q_t_[i] * GAMMA))
                y = Variable(torch.FloatTensor(y))
                # update parameter
                loss = dqn.loss_func(q_j, y)
                tmp += float(loss[0])
                tmp_r += GAMMA**t * r
                dqn.optimizer.zero_grad()
                loss.backward()
                dqn.optimizer.step()
                if done:
                    print('Ep: ', i_episode, '| Ep_r: ', t, '| Epsilon: ', EPSILON)
                    break
            if done:
                break
            s = s_
            cnt += 1
        point.append(tmp / cnt)
        point_r.append(tmp_r / cnt)
        print(t, i_episode, EPSILON)
        if EPSILON > MIN_EPSILON:
            EPSILON -= 0.01
        #EPSILON = get_explore_rate(i_episode)
    return point, point_r, dqn.net, env

def run_episode(env, net=None, render=False):
    obs = env.reset()
    obs = Variable(torch.unsqueeze(torch.FloatTensor(obs), 0))
    total_reward = 0
    step_idx = 0
    for _ in range(MAX_T):
        if render:
            env.render()
        if net is None:
            action = env.action_space.sample()
        else:
            actions_value = net.forward(obs)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        obs, reward, done, _ = env.step(action)
        obs = Variable(torch.unsqueeze(torch.FloatTensor(obs), 0))
        total_reward += GAMMA ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward


if __name__ == "__main__":
    p, p_r, qnet, env = learn('Acrobot-v1')
    #env = wrappers.Monitor(env, '/tmp/gym/moutaincar-experiment-2', force=True)
    print(p)
    l = len(p)
    x = [i for i in range(l)]
    solution_policy_scores = [run_episode(env, qnet, False) for _ in range(50)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    print("Standard Deviation = ", np.std(solution_policy_scores))
    #run_episode(env, qnet, True)

    plt.plot(x, p, marker='', mec='r', mfc='w', label='loss')
    plt.plot(x, p_r, marker='', ms=10, label='reward')
    plt.legend()  # 让图例生效
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    env.close()
    # learn('MountainCar-v0')