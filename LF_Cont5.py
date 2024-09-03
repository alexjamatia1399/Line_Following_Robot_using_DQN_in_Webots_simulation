import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from controller import Supervisor

# DQN Architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# Constants
TIME_STEP = 64
MAX_SPEED = 3
TRAIN_EPISODES = 10000

# Create the Supervisor instance
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Initialize devices (ground sensors and motors)
gs = []
gsNames = ['gs0', 'gs1', 'gs2']
for i in range(3):
    gs.append(robot.getDevice(gsNames[i]))
    gs[i].enable(timestep)

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Convert sensor readings to state strings
def sensors_to_state(values):
    return ''.join(['1' if v > 600 else '0' for v in values])

# Reward matrix based on state transitions
reward_matrix = {
    '000': {'000': -1, '001': 1, '010': 1, '011': 11, '100': 1, '101': 11, '110': 11, '111': 21},
    '001': {'000': -10, '001': 1, '010': 1, '011': 2, '100': 1, '101': 11, '110': 11, '111': 11},
    '010': {'000': -10, '001': 1, '010': 1, '011': 2, '100': 1, '101': 11, '110': 11, '111': 11},
    '011': {'000': -20, '001': -10, '010': -10, '011': 1, '100': 1, '101': 2, '110': 2, '111': 10},
    '100': {'000': -10, '001': 1, '010': 1, '011': 11, '100': 1, '101': 11, '110': 11, '111': 11},
    '101': {'000': -20, '001': -10, '010': -10, '011': 1, '100': -10, '101': 1, '110': 2, '111': 10},
    '110': {'000': -20, '001': -10, '010': -10, '011': 1, '100': -10, '101': 2, '110': 1, '111': 10},
    '111': {'000': -30, '001': -20, '010': -20, '011': -10, '100': -20, '101': -10, '110': -10, '111': 50}
}

# Training the DQN
def train_dqn(episodes):
    policy_net = DQN(3, 3)
    target_net = DQN(3, 3)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    buffer = ReplayBuffer(2000)
    epsilon = 1.0

    for episode in range(episodes):
        print(f"Starting Episode {episode+1}")

        # Reset the speed of the robot at the start of a new episode
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)

        gsValues = [gs[i].getValue() for i in range(3)]
        state = np.array(gsValues)
        current_state = sensors_to_state(gsValues)
        done = False
        off_line_counter = 0

        while robot.step(timestep) != -1 and not done:
            print(f"GS0: {gsValues[0]}, GS1: {gsValues[1]}, GS2: {gsValues[2]}, Episode: {episode+1}")

            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state).float().unsqueeze(0))
                    action = q_values.argmax().item()

            leftSpeed = 0
            rightSpeed = 0
            if action == 0:
                leftSpeed = MAX_SPEED
                rightSpeed = MAX_SPEED
            elif action == 1:
                leftSpeed = MAX_SPEED / 2
                rightSpeed = MAX_SPEED
            elif action == 2:
                leftSpeed = MAX_SPEED
                rightSpeed = MAX_SPEED / 2

            leftMotor.setVelocity(leftSpeed)
            rightMotor.setVelocity(rightSpeed)
            robot.step(timestep)

            next_gsValues = [gs[i].getValue() for i in range(3)]
            next_state = np.array(next_gsValues)
            next_state_str = sensors_to_state(next_gsValues)

            num_sensors_off_line = sum(1 for value in next_gsValues if value > 600)
            if num_sensors_off_line == 3:
                print("Robot is completely off the line.")
                off_line_counter += 1
            elif num_sensors_off_line == 2:
                print("Robot has two sensors off the line.")
                off_line_counter += 0.3
            elif num_sensors_off_line == 1:
                print("Robot has one sensor off the line.")
                off_line_counter += 0.6
            else:
                print("Robot is on the line.")
                off_line_counter = 0

            reward = reward_matrix[current_state].get(next_state_str, -1)

            if off_line_counter * timestep / 1000 >= 1:
                done = True
                print("Resetting the robot to its initial state.")
                translationField = robot.getSelf().getField('translation')
                rotationField = robot.getSelf().getField('rotation')
                translationField.setSFVec3f([0.129027, -0.462746, -9.2752e-05])
                rotationField.setSFRotation([0.000406784, -2.8794e-05, 1, 3.04359])
                print("Robot position and orientation reset.")

            buffer.push(state, action, reward, next_state, done)

            if len(buffer) >= 128:
                states, actions, rewards, next_states, dones = buffer.sample(128)
                states = torch.tensor(states).float()
                actions = torch.tensor(actions).long()
                rewards = torch.tensor(rewards).float()
                next_states = torch.tensor(next_states).float()
                dones = torch.tensor(dones).float()

                current_q_values = policy_net(states).gather(1, actions.unsqueeze(-1))
                next_q_values = target_net(next_states).max(1)[0].detach()
                target_q_values = rewards + (0.99 * next_q_values * (1 - dones))
                loss = nn.functional.mse_loss(current_q_values, target_q_values.unsqueeze(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            current_state = next_state_str
            epsilon = max(0.01, epsilon * 0.998)
            gsValues = next_gsValues

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net.state_dict(), 'dqn_model.pth')

train_dqn(TRAIN_EPISODES)
