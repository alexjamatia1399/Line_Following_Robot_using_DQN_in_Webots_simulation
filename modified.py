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
MAX_SPEED = 6.28
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

# Training the DQN (adapted for the e-puck robot)
def train_dqn(episodes):
    policy_net = DQN(3, 3)
    target_net = DQN(3, 3)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    buffer = ReplayBuffer(1000)
    epsilon = 1.0

    for episode in range(episodes):
        print(f"Starting Episode {episode+1}")
        
        # Reset the speed of the robot at the start of a new episode
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)

        gsValues = [gs[i].getValue() for i in range(3)]
        state = np.array(gsValues)
        done = False
        off_line_counter = 0

        while robot.step(timestep) != -1 and not done:
            print(f"GS0: {gsValues[0]}, GS1: {gsValues[1]}, GS2: {gsValues[2]}, Episode: {episode+1}")

            # Determine action based on epsilon-greedy policy
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state).float().unsqueeze(0))
                    action = q_values.argmax().item()

            # Set motor speeds based on the action
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

            next_gsValues = [gs[i].getValue() for i in range(3)]
            next_state = np.array(next_gsValues)

            # Reward mechanism based on the number of sensors off the line
            num_sensors_off_line = sum(1 for value in next_gsValues if value > 600)
            if num_sensors_off_line == 3:
                print("Robot is completely off the line.")
                off_line_counter += 1
                reward = -1.0
            elif num_sensors_off_line == 2:
                print("Robot has two sensors off the line.")
                off_line_counter += 0.3
                reward = 0.3
            elif num_sensors_off_line == 1:
                print("Robot has one sensor off the line.")
                off_line_counter += 0.6
                reward = 0.6
            else:
                print("Robot is on the line.")
                off_line_counter = 0
                reward = 1.0

            # Check if the robot needs to be reset
				print(f"{off_line_counter}")
				print(f"{timestep}")
            if off_line_counter * timestep / 1000 >= .7:
                done = True
                print("Resetting the robot to its initial state.")
                translationField = robot.getSelf().getField('translation')
                rotationField = robot.getSelf().getField('rotation')
                translationField.setSFVec3f([0.23064, -0.446194, -8.84811e-05])
                rotationField.setSFRotation([0.000304348, -3.93093e-05, -1, 3.10191])
                print("Robot position and orientation reset.")

            buffer.push(state, action, reward, next_state, done)

            # Experience replay and model update
            if len(buffer) >= 64:
                states, actions, rewards, next_states, dones = buffer.sample(64)
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
            epsilon = max(0.01, epsilon * 0.995)
            gsValues = next_gsValues

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net.state_dict(), 'dqn_model.pth')

train_dqn(TRAIN_EPISODES)
