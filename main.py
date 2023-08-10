import gymnasium
import time
from nn_model import NeuralNetwork
import numpy as np

TRAINING = False

env         = gymnasium.make("InvertedDoublePendulum-v4", render_mode='human')
observation = env.reset()
done        = False
trials      = 0
reward      = None

nn = NeuralNetwork()
nn.compile()

all_trials_x = []
all_trials_y = []


input_vector = np.zeros(12)  # Initialize with 32 zeros
action = [0]
idx = 0

while trials <= 1000:
    observation, reward, done, info, _ = env.step(action)
    action = env.action_space.sample()

    input_vector[0]  = observation[0]
    input_vector[1]  = observation[1]
    input_vector[2]  = observation[2]
    input_vector[3]  = observation[3]
    input_vector[4]  = observation[4]
    input_vector[5]  = observation[5]    
    input_vector[6]  = observation[6]
    input_vector[7]  = observation[7]
    input_vector[8]  = observation[8]
    input_vector[9]  = observation[9]
    input_vector[10] = observation[10]
    input_vector[11] = action

    # Save the increase in reward between the current reward and the previous reward
    all_trials_y.append([reward])
    all_trials_x.append(input_vector)
    idx += 1
    if done or idx >= 32:
        
        print(reward)
        trials += 1
        env.reset()
        input_vector = np.zeros(12) 
        idx = 0

env.close()