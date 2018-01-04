import gym
from gym import wrappers
import numpy as np
import random
import math

# parameter
MAX_EPISODE = 1000
#MAX_T = 20000
MAX_T = 2000
#T_SLOVE_JUDGE = 195
#E_SLOVE_JUDGE = 100
#EPSILON=0.5
#EPSILON_DECAY=0.99
#ALPHA=0.9
GAMMA=0.99
MIN_EXPLORE_RATE = 0.01
#MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
DEBUG_MODE = False
'''
global env
global q
global state_space_dis
global  split_n
'''
#function
def get_state(state_space_dis, split_n, state):
    discrete = []
    l = len(state)
    for i in range(l):
        if state[i] <= state_space_dis[i][0]:
            index = 0
        elif state[i] >= state_space_dis[i][1]:
            index = split_n[i] - 1
        else:
            width = (state_space_dis[i][1] - state_space_dis[i][0]) / split_n[i]
            index = math.floor((state[i] - state_space_dis[i][0]) / width)
        discrete.append(index)
    return tuple(discrete)

def get_action(env, state, q, rate):
    if random.random() < rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(q[state])
    return action

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

#start
def qlearning(model):
    # select environment
    env = gym.make(model)
    env = env.unwrapped
    n_action = env.action_space.n

    # state discretization
    if model == 'CartPole-v0':
        state_min = env.observation_space.low
        state_min[1], state_min[3] = -0.5, -1
        state_max = env.observation_space.high
        state_max[1], state_max[3] = 0.5, 1
        split_n = (1, 1, 6, 3)
    elif model == "MountainCar-v0":
        state_min = env.observation_space.low
        state_max = env.observation_space.high
        split_n = (40, 40)
    else:
        state_min = env.observation_space.low
        state_max = env.observation_space.high
        split_n = (10, 10, 10, 10, 10, 10)
    state_space_dis = list(zip(state_min, state_max))

    # initialize
    q = np.zeros(split_n + (n_action,))

    #start
    e = 0
    EPSILON = get_explore_rate(0)
    ALPHA = get_learning_rate(0)
    for i in range(MAX_EPISODE):
        obs = env.reset()
        state = get_state(state_space_dis, split_n, obs)
        for t in range(MAX_T):
            #env.render()
            action = get_action(env, state, q, EPSILON)
            obs, reward, done, info = env.step(action)
            state_n = get_state(state_space_dis, split_n, obs)
            q_best = np.amax(q[state_n])
            q[state + (action,)] += ALPHA * (reward + GAMMA * (q_best) - q[state + (action,)])
            state = state_n
            #debug
            if (DEBUG_MODE):
                print("\nEpisode = %d" % i)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % q_best)
                print("Explore rate: %f" % EPSILON)
                print("Learning rate: %f" % ALPHA)
                print("Streaks: %d" % e)

                print("")
            if done:
                print("Episode %d finished after %f time steps" % (i, t))
                '''
                if (t >= T_SLOVE_JUDGE):
                    e += 1
                else:
                    e = 0
                    '''
                break

        #if e > E_SLOVE_JUDGE:
            #break
        #EPSILON = EPSILON * EPSILON_DECAY
        EPSILON = get_explore_rate(i)
        ALPHA = get_learning_rate(i)

    #print(q)
    #print('split')
    return env, q

def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    state_min = env.observation_space.low
    #state_min[1], state_min[3] = -0.5, -1
    state_max = env.observation_space.high
    #state_max[1], state_max[3] = 0.5, 1
    #split_n = (1, 1, 6, 3)
    #split_n = (40, 40)
    split_n = (10, 10, 10, 10, 10, 10)
    state_space_dis = list(zip(state_min, state_max))
    for _ in range(MAX_T):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            state = get_state(state_space_dis, split_n, obs)
            action = policy[state]
        obs, reward, done, _ = env.step(action)
        total_reward += GAMMA ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

if __name__ == "__main__":
    env, q_table = qlearning('Acrobot-v1')
    #env = wrappers.Monitor(env, '/tmp/gym/moutaincar-experiment-1', force=True)
    print(q_table)
    print(q_table.shape)
    #solution_policy = np.argmax(q_table, axis=4)
    #solution_policy = np.argmax(q_table, axis=2)
    solution_policy = np.argmax(q_table, axis=6)
    print("split")
    print(solution_policy.shape)
    print(solution_policy)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    print("Standard Deviation = ", np.std(solution_policy_scores))
    #run_episode(env, solution_policy, True)
    #env.close()

