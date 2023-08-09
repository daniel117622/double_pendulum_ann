import gymnasium
import time
env = gymnasium.make("InvertedDoublePendulum-v4", render_mode='human')
observation = env.reset()
done = False

while True:
    action = env.action_space.sample()  
    observation, reward, done, info, _= env.step(action)
    env.render()
    time.sleep(0.5)
    if done:
        env.reset()
    print(f"Pole-Cart angle :{observation[1]*90:.3f}")
    

env.close()