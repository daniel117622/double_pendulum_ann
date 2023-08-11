import gymnasium as gym
import numpy as np

# Crear el entorno CartPole
env = gym.make('CartPole-v1')

# Parámetros del algoritmo Q-learning
num_episodes = 1000
max_steps_per_episode = 1000
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 1.0
min_exploration_prob = 0.01
exploration_decay = 0.995

# Crear tabla Q
num_states = np.prod(np.array(env.observation_space.shape))
num_actions = env.action_space.n
q_table = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

# Función para discretizar el espacio de estados
def discretize_state(state):
    return tuple(np.round(state, decimals=1))

# Entrenamiento con Q-learning
idx = 0
for episode in range(num_episodes):
    state = env.reset()
    if idx != 0:
        state = discretize_state(state)
    else:
        state = discretize_state(state[0])
    for step in range(max_steps_per_episode):
        # Selección de acción utilizando epsilon-greedy
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Tomar la acción y obtener la nueva información del entorno
        next_state, reward, done, _ , _= env.step(action)
        next_state = discretize_state(next_state)

        # Actualizar la tabla Q utilizando la ecuación de Q-learning
        if done:
            q_table[state][action] = (1 - learning_rate) * q_table[state][action] + \
                                     learning_rate * (reward - q_table[state][action])
        else:
            q_table[state][action] = (1 - learning_rate) * q_table[state][action] + \
                                     learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action])

        state = next_state

        if done:
            break

    # Decrementar la probabilidad de exploración
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)

# Evaluar el agente entrenado
num_eval_episodes = 10
for episode in range(num_eval_episodes):
    state = env.reset()
    state = discretize_state(state)
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)
        total_reward += reward

        if done:
            print(f"Episode {episode + 1}, Steps: {step + 1}, Total Reward: {total_reward}")
            break
        state = next_state

env.close()