import gymnasium
import time
from nn_model import NeuralNetwork

env         = gymnasium.make("InvertedDoublePendulum-v4", render_mode='human')
observation = env.reset()
done        = False
trials      = 0

nn = NeuralNetwork()

print(env.action_space.sample())
action = [0]
while trials <= 1000:
    observation, reward, done, info, _ = env.step(action)
    action = [-observation[5]]
    env.render()
    time.sleep(0.05)
    if done:
        trials += 1
        env.reset()
    print(f"Pole-Cart angle :{observation[1]*90:.3f}")
    
env.close()