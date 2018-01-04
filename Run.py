import sys
import MyQLearning
import MyDQN
import MyImprovedDQN

# implement q learning
if sys.argv[1] == 'q1':
    MyQLearning.qlearning('CartPole-v0')
if sys.argv[1] == 'q2':
    MyQLearning.qlearning('MountainCar-v0')
if sys.argv[1] == 'q3':
    MyQLearning.qlearning('Acrobot-v1')

# implement dqn
if sys.argv[1] == 'dq1':
    MyDQN.qlearning('CartPole-v0')
if sys.argv[1] == 'dq2':
    MyDQN.qlearning('MountainCar-v0')
if sys.argv[1] == 'dq3':
    MyDQN.qlearning('Acrobot-v1')

# implement improved dqn
if sys.argv[1] == 'dqi1':
    MyImprovedDQN.learn('CartPole-v0')
if sys.argv[1] == 'dqi2':
    MyImprovedDQN.learn('MountainCar-v0')
if sys.argv[1] == 'dqi3':
    MyImprovedDQN.learn('Acrobot-v1')

