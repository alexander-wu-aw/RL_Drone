#!/usr/bin/python
import numpy as np
import tensorflow as tf
import gym
import random
import copy
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import roslib
import rospy
import random
import time
import math
from std_srvs.srv import Empty as empty
from std_msgs.msg import Empty
from gazebo_msgs.srv import SetModelConfiguration
from gazebo_msgs.srv import GetModelState

from control_msgs.msg import JointControllerState
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from std_msgs.msg import Float64
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

rospy.init_node('control_script')
rate = rospy.Rate(120)
print 'initiated'

pub_takeoff = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=10)
pub_land = rospy.Publisher('/ardrone/land', Empty, queue_size=10)
pub_action = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', empty)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', empty)

# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_length):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(in_features = input_length, out_features = 128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=32)
        self.out = nn.Linear(in_features = 32, out_features=4)
    
    # Forward pass
    def forward(self, t):
        t = torch.FloatTensor(t)
        #t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = F.relu(self.fc4(t))
        t = self.out(t)
        return t

# Exploration Rate
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    
    # Get range of exploration rate max and min
    # multiply by decay rate: current step times decay rate
    # Add to exploration min
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
                math.exp(-1. * current_step * self.decay)


Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward' ))

# Replay Memory
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity # Set the max capacity of memory
        self.memory = [] # Array to store experiences
        self.push_count = 0 # track how many experiences we pushed
        
    def push(self,experience):
        # If memory still not at capacity
        if len(self.memory) < self.capacity:
            # Put experience in memory
            self.memory.append(experience)
        else:
            # If memory already full
            # Replace the oldest experience with the experience
            self.memory[self.push_count % self.capacity] = experience
        # Increment push count
        self.push_count += 1
    
    # Get random batch of memories
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    # Make sure there are enough experiences in memory
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class Eval_mode():
    def get_exploration_rate(self, current_step):
        return 0

class PIDController(object):
    def __init__(self, kp, kd=0, ki=0):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_error_z = 0.0
        self.last_error_rot_z = 0.0
        self.last_iterm_x = 0.0
        self.last_iterm_y = 0.0
        self.last_iterm_z = 0.0
        self.last_iterm_rot_z = 0.0
        self.time_delta = 0.00833

    def run(self, current_x, goal_x, current_y, goal_y, current_z, goal_z, current_rot_z, goal_rot_z):
        error_x = goal_x - current_x
        error_x_delta = error_x - self.last_error_x
        self.last_iterm_x += error_x_delta*self.time_delta
        u_x = self.kp*error_x + (self.kd*error_x_delta)/self.time_delta + self.ki*self.last_iterm_x
        self.last_error_x = error_x

        error_y = goal_y - current_y
        error_y_delta = error_y - self.last_error_y
        self.last_iterm_y += error_y_delta*self.time_delta
        u_y = self.kp*error_y + (self.kd*error_y_delta)/self.time_delta + self.ki*self.last_iterm_y
        self.last_error_y = error_y

        error_z = goal_z - current_z
        error_z_delta = error_z - self.last_error_z
        self.last_iterm_z += error_z_delta*self.time_delta
        u_z = self.kp*error_z + (self.kd*error_z_delta)/self.time_delta + self.ki*self.last_iterm_z
        self.last_error_z = error_z

        kpr = 0.5; kdr = 0.001; kir = 0.01
        error_rot_z = goal_rot_z - current_rot_z
        error_rot_z_delta = error_rot_z - self.last_error_rot_z
        self.last_iterm_rot_z += error_rot_z_delta*self.time_delta
        u_rot_z = kpr*error_rot_z + (kdr*error_rot_z_delta)/self.time_delta + kir*self.last_iterm_rot_z
        self.last_error_rot_z = error_rot_z
        
        #print error_x, ", ", error_y, ", ", error_z, ", "

        return u_x, u_y, u_z, u_rot_z

pid = PIDController(4.75, 0.25, 0.75)

# Agent
class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0 # Current step number
        self.strategy = strategy # Exploration rate strategy
        self.num_actions = num_actions # Number of actions I can take
        self.device = device # Device to move to

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.x_dot = 0.0
        self.y_dot = 0.0
        self.z_dot = 0.0
        self.drone_state = [self.x, self.y, self.z, self.x_dot, self.y_dot, self.z_dot]
        self.done = False
        self.is_reset = False
        self.Q = dict()
        self.epsilon = 1.0
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.alpha = 0.8
        self.gamma = 0.9
    
    # Select an action to take
    def select_action(self, state, policy_net, device):
        # Get exploration rate
        rate = strategy.get_exploration_rate(self.current_step)
        # Increment step count
        self.current_step += 1
        
        # If explore
        if rate > random.random():
            # Get action by getting random value
            action = random.randrange(self.num_actions)
            print("Action:",action)
            return torch.tensor([action]).to(device) # explore
        else:
            with torch.no_grad():
                # Pass state through policy network
                # Get the index of maximum value
                q_values = policy_net(state)
                print("Qvalues:", q_values,"Action:", q_values.argmax().to(device) )
                return q_values.argmax().to(device) # exploit
            
    def current_step(self):
        return self.current_step
    
    def set_current_step(self, step):
        self.current_step = step

    def set_drone_state(self, state):
        self.drone_state = [state[0], state[1], state[2], state[3], state[4], state[5]]

    def get_drone_state(self):
        return self.drone_state

    def reset(self):
        rospy.wait_for_service('gazebo/reset_world')
        try:
            reset_simulation()
        except(rospy.ServiceException) as e:
            print "reset_world failed!"

        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     pause()
        # except (rospy.ServiceException) as e:
        #     print "rospause failed!"

        print "called reset()"
        return env.get_drone_state()

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            unpause()
        except (rospy.ServiceException) as e:
            print "/gazebo/pause_physics service call failed"

        command = Twist()
        
        max_vel = 1.0
        
        current_state = env.get_drone_state()
        current_state_x = int(round(current_state[0]))
        current_state_y = int(round(current_state[1]))
        current_state_z = int(current_state[2])
        current_state_rot_z = int(env.rot_z)
        goal_state_rot_z = 0.0
        goal_state_x = 0
        goal_state_y = current_state_y
        goal_state_z = 1.0

        pid.last_error_x = 0.0
        pid.last_error_y = 0.0
        pid.last_error_z = 0.0
        pid.last_error_rot_z = 0.0

        if action == 0:
            # Forward
            goal_state_x = current_state_x + 1
            goal_state_y = current_state_y
            old_x = 0.0;
            while True:
                if old_x != current_state_x:
                    # print 'boom!'
                    old_x = current_state_x
                    u_x, u_y, u_z, u_rot_z = pid.run(current_state_x, goal_state_x, current_state_y, goal_state_y, current_state_z, goal_state_z, current_state_rot_z, goal_state_rot_z)                
                    command.linear.x = u_x
                    command.linear.y = u_y
                    command.linear.z = u_z
                    command.angular.x = 0.0
                    command.angular.y = 0.0
                    command.angular.z = u_rot_z
                    pub_action.publish(command)
                    if abs(goal_state_x - current_state_x ) <= 0.05 and abs(goal_state_y - current_state_y) <= 0.05 \
                    and abs(goal_state_z - current_state_z) <= 0.05 and abs(goal_state_rot_z - current_state_rot_z ) <= 0.1:
                        break
                current_state = env.get_drone_state()
                current_state_x = (current_state[0])
                current_state_y = (current_state[1])
                current_state_z = current_state[2]
                current_state_rot_z = env.rot_z
                rate.sleep()

        if action == 1:
            # Backwards
            goal_state_x = current_state_x - 1
            goal_state_y = current_state_y
            old_x = 0.0;
            while True:
                if old_x != current_state_x:
                    # print 'boom!'
                    old_x = current_state_x
                    u_x, u_y, u_z, u_rot_z = pid.run(current_state_x, goal_state_x, current_state_y, goal_state_y, current_state_z, goal_state_z, current_state_rot_z, goal_state_rot_z)
                    command.linear.x = u_x
                    command.linear.y = u_y
                    command.linear.z = u_z
                    command.angular.x = 0.0
                    command.angular.y = 0.0
                    command.angular.z = u_rot_z
                    pub_action.publish(command)
                    if abs(goal_state_x - current_state_x ) <= 0.05 and abs(goal_state_y - current_state_y) <= 0.05 \
                    and abs(goal_state_z - current_state_z) <= 0.05 and abs(goal_state_rot_z - current_state_rot_z ) <= 0.1:
                        break
                current_state = env.get_drone_state()
                current_state_x = (current_state[0])
                current_state_y = (current_state[1])
                current_state_z = current_state[2]
                current_state_rot_z = env.rot_z
                rate.sleep()

        if action == 2: 
            # Left
            goal_state_y = current_state_y + 1
            goal_state_x = current_state_x
            old_x = 0.0;
            while True:
                if old_x != current_state_x:
                    # print 'boom!'
                    old_x = current_state_x
                    u_x, u_y, u_z, u_rot_z = pid.run(current_state_x, goal_state_x, current_state_y, goal_state_y, current_state_z, goal_state_z, current_state_rot_z, goal_state_rot_z)
                    command.linear.x = u_x
                    command.linear.y = u_y
                    command.linear.z = u_z
                    command.angular.x = 0.0
                    command.angular.y = 0.0
                    command.angular.z = u_rot_z
                    pub_action.publish(command)
                    if abs(goal_state_y - current_state_y ) <= 0.05 and abs(goal_state_x - current_state_x ) <= 0.05 \
                    and abs(goal_state_z - current_state_z) <= 0.05 and abs(goal_state_rot_z - current_state_rot_z ) <= 0.1:   
                        break
                current_state = env.get_drone_state()
                current_state_x = (current_state[0])
                current_state_y = (current_state[1])
                current_state_z = current_state[2]
                current_state_rot_z = env.rot_z
                rate.sleep()

        if action == 3: 
            #Right
            goal_state = current_state_y - 1
            goal_state_x = current_state_x
            old_x = 0.0;
            while True:
                if old_x != current_state_x:
                    # print 'boom!'
                    old_x = current_state_x
                    u_x, u_y, u_z, u_rot_z = pid.run(current_state_x, goal_state_x, current_state_y, goal_state_y, current_state_z, goal_state_z, current_state_rot_z, goal_state_rot_z)
                    command.linear.x = u_x
                    command.linear.y = u_y
                    command.linear.z = u_z
                    command.angular.x = 0.0
                    command.angular.y = 0.0
                    command.angular.z = u_rot_z
                    pub_action.publish(command)
                    if abs(goal_state_y - current_state_y ) <= 0.05 and abs(goal_state_x - current_state_x ) <= 0.05 \
                    and abs(goal_state_z - current_state_z) <= 0.05 and abs(goal_state_rot_z - current_state_rot_z ) <= 0.1:   
                        break
                current_state = env.get_drone_state()
                current_state_x = (current_state[0])
                current_state_y = (current_state[1])
                current_state_z = current_state[2]
                current_state_rot_z = env.rot_z
                rate.sleep()

        next_state = env.get_drone_state()
        reward = get_reward(next_state)
        
        if (next_state[0] < -0.05 or next_state[0] > 5.05 or next_state[1] < -0.05 or next_state[1] > 5.05 or reward == 100):
            env.done = True
            reward = reward - 100

        return next_state, reward, env.done

def get_reward(state):
    reward = -3*(abs(state[0] - 5.0)**2 + abs(state[1] - 5.0)**2) 
    if abs(state[0] - 5.0) <= 0.05 and abs(state[1] - 5.0) <= 0.05:
        # goal reached 
        reward = 1000
    return reward

def takeoff():
    pub_takeoff.publish(Empty())

def land():
    pub_land.publish(Empty())

def cb_drone_state(data):
#    print("TESTING",data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z)
    env.set_drone_state([data.pose.pose.position.x,data.pose.pose.position.y, data.pose.pose.position.z, data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z])

def cb_drone_navdata(data):
#    print("TEST!!", data.rotX, data.rotY, data.rotZ)
    env.rot_x = data.rotX
    env.rot_y = data.rotY
    env.rot_z = data.rotZ
    
def subscriber():
    rospy.Subscriber("/ground_truth/state", Odometry, cb_drone_state)
    rospy.Subscriber("/ardrone/navdata", Navdata, cb_drone_navdata)

def plot(values,value2, moving_avg_period, epsilon):
    # Set up figure
    plt.figure(2)
    plt.clf()
    # Give title
    plt.title('Training...')
    # Name axis
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    # Give values to plot - episode durations
    plt.plot(values)
    plt.plot(value2)
    
    moving_avg = get_moving_average(moving_avg_period, values)
    # Plot moving average
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", \
          moving_avg_period, "episode moving avg:", moving_avg[-1], \
         "\n", "Exploration rate (epsilon): ", epsilon)
    #if is_ipython: display.clear_output(wait=True)

# Get moving average of the episode duration
def get_moving_average(period,values):
    # Convert to pytorch tensor
    values = torch.tensor(values, dtype=torch.float)
    # If there are enough values in dataset to fill a moving average period
    # Since we can't calculate moving average if length of dataset is less than moving average period
    if len(values) >= period:
        # Unfold: return tensor that contains all slices with size = period on 0th dimension
        # Mean: mean of each slice
        # Flatten: flatten tensor
        # Now: tensor of all periods moving averages from values
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
                        .mean(dim=1).flatten(start_dim=0)
        # Moving average for first (period-1) values will be 0
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy() # convert to numpy array
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy() # Numpy array of all zeros


def extract_tensors(experiences):
    
    # Transpose to batch of experience
    # EG Instead of three seperate Experience instances, have one instance with 3 values in each tuple
    # This gives one experience instance, with each state, action, next_state, reward having a tuple of all values in batch
    batch = Experience(*zip(*experiences))
    
    # Turn each into their own tensor
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    
    return (t1, t2, t3, t4)

class QValues():
    # Since we do not set instance of this class, we need to set device here locally
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        # Returns predicted Q-values for state-action pairs
        #return policy_net(states).gather(dim =1, index=actions.unsqueeze(-1))
        return policy_net(states)
    
    @staticmethod
    # Get the maximum Q-value predicted by target net among all possible next actions
    def get_next(target_net, next_states):
        return target_net(next_states)

# Main Program
# Hyperperameters
batch_size = 256 # batch size
gamma = 0.999 # Discount factor in bellman equaiton
eps_start = 1 # Exploration rate
eps_end = 0.01 # Epsilon
eps_decay = 0.01 #0.001
target_update = 10 # How frequently we update target network with policy network weights (in episodes)
memory_size = 100000 # How many expereinces will remember
lr = 0.4 # Learning rate
num_episodes = 1000 # Num of episodes

# USe GPU is available, otherwise, cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set strategy with epsilon greedy strategy
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
# Define agent 
env = Agent(strategy, 4, device)
# initialize memory
memory = ReplayMemory(memory_size)

num_episodes = 10000

subscriber()

rospy.sleep(1)

state = env.reset()

print("state", state)

# Set up policy network - input shape is height and width of screen
policy_net = DQN(len(state)).to(device)
# Set up target network
target_net = DQN(len(state)).to(device)


###
# Load parameters that I trained before
#state_dict = torch.load('../ardrone_ws/src/rl/src/model.pt')
#policy_net.load_state_dict(state_dict)

#env.set_current_step(torch.load('../ardrone_ws/src/rl/src/step.pt'))
###


# Set weights and biases of target net = policy net
target_net.load_state_dict(policy_net.state_dict())
# Put target net in eval mode - which means its NOT in training mode
# Only used for inference
target_net.eval()
# Set up Adam optimizer
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

takeoff()
rospy.sleep(3.0)

# Store episode durations
episode_durations = []
episode_losses = []
# Loop through episodes
for episode in range(num_episodes):
    state = env.reset()
    env.done = False
    total_reward = 0
    total_losses = 0
    step_count = 0
    rospy.sleep(0.25)
    while env.done == False:
        action = env.select_action(state, policy_net, device)
                    
        #take an e-greedy action
        next_state, reward, done = env.step(int(action))
        step_count +=1
        print("reward: ", reward)
        total_reward += reward
        
        current_q_values = QValues.get_current(policy_net, state, action)
        # Get Q-value for next state in batch - using target_net
        next_q_values = QValues.get_next(target_net, next_state)
        # Calculate target Q value using formula
        target_q_values = (next_q_values * gamma) + reward
        
        # Get loss between current and target Q values
        loss = F.mse_loss(current_q_values, target_q_values)
        print("loss: ", loss.data)
        total_losses += loss
        # Set gradients of all wieghts and biases in policy net to 0
        optimizer.zero_grad()
        # Compute gradient of loss for policy net
        loss.backward()
        # Updates weights and loss using gradients calculated above
        optimizer.step()

        state = next_state

    print "reward in episode ", episode," is: ",total_reward
    # Append timestep to list
    episode_durations.append(total_reward)
    episode_losses.append(total_losses/(100*step_count))
    # Plot it
    plot(episode_durations,episode_losses, 100, strategy.get_exploration_rate(env.current_step))


    '''
    
    # Iterate over each timestep in the episode
    for timestep in count():
        # Select action based on state - using policy_net
        action = env.select_action(state, policy_net, device)
        # Take action and get reward
        next_state, reward, done = env.step(int(action))
        #reward = em.step(action)
        # Get the next state
        #next_state = em.get_state()
        # Create experience and push it into replay memory
        memory.push(Experience(state,action,next_state,reward))
        # Update current state to next state
        state = next_state
        
        # See if we can get a sample from replay memory to train policy net
        # If enough experiences - more than batch size
        if memory.can_provide_sample(batch_size):
            # Get a sample
            experiences = memory.sample(batch_size)
            # Extract state, actions, rewards and next state into their own tensors
            states, actions, rewards, next_states = extract_tensors(experiences)
            
            # Get Q-values for state-action pairs from batch as pytorch tensor
            current_q_values = QValues.get_current(policy_net, states, actions)
            # Get Q-value for next state in batch - using target_net
            next_q_values = QValues.get_next(target_net, next_states)
            # Calculate target Q value using formula
            target_q_values = (next_q_values * gamma) + rewards
            
            # Get loss between current and target Q values
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            # Set gradients of all wieghts and biases in policy net to 0
            optimizer.zero_grad()
            # Compute gradient of loss for policy net
            loss.backward()
            # Updates weights and loss using gradients calculated above
            optimizer.step()
        
        # Check if agent had action that ended episode
        if env.done:
            # Append timestep to list
            episode_durations.append(timestep)
            # Plot it
            plot(episode_durations, 100, strategy.get_exploration_rate(env.current_step))
            # Break out of inner loop to start new episode
            break
    '''
        
    # Check if we should update target net
    if episode % target_update ==0:
        # Update target net to match policy net
        print("Updating target net")
        target_net.load_state_dict(policy_net.state_dict())
    
    if episode > 9 and episode % 10 == 0:
        # Save model
        print("Saving model")
        torch.save(policy_net.state_dict(), '../ardrone_ws/src/rl/src/model.pt')
        # Save Exploration rate (epsilon)
        torch.save(env.current_step, '../ardrone_ws/src/rl/src/step.pt')



