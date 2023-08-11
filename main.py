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

while trials <= 100:
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
    input_vector[11] = reward

    # Save the increase in reward between the current reward and the previous reward
    all_trials_y.append(action)
    all_trials_x.append(input_vector)
    idx += 1
    if done or idx >= 32:
        trials += 1
        env.reset()
        input_vector = np.zeros(12) 
        idx = 0


all_trials_x  =  np.array(all_trials_x)
all_trials_y  =  np.array(all_trials_y)

nn.train(all_trials_x, all_trials_y)
all_trials_x[0][11] = 10
action = nn.inference(all_trials_x[0])
print(action)

trials = 0
while trials <= 100:
    observation, reward, done, info, _ = env.step([action])
    state = [ *observation , 10 ] 
    action = [nn.inference(state)]
    print(action)
    time.sleep(0.25)
    if done:
        trials += 1
        env.reset()
        input_vector = np.zeros(12) 
        idx = 0

