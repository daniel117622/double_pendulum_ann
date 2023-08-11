from scipy.integrate import odeint
import numpy as np
import random

class InvertedPendulum:
    def __init__(self, mass_cart, mass_pendulum, length, gravity=9.81):
        self.mass_cart = mass_cart
        self.mass_pendulum = mass_pendulum
        self.length = length
        self.gravity = gravity
        self.force = 0.0
        self.state = np.array([0.0, 0.0, 0.0, 0.0])  # x, x_dot, theta, theta_dot


    def add_force(self, force):
        self.force = force

    def get_state(self):
        return self.state
    
    def loss(self):
        # Try to minimize the velocity, and the angular movements.
        eq = self.state[1] + abs(self.state[2]) + abs(self.state[3])
        return eq

def equations_of_motion(state, t, inverted_pendulum):
    m_c = inverted_pendulum.mass_cart
    m_p = inverted_pendulum.mass_pendulum
    l = inverted_pendulum.length
    g = inverted_pendulum.gravity
    F = inverted_pendulum.force
    
    x, x_dot, theta, theta_dot = state

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denominator = m_c + m_p * sin_theta ** 2

    x_ddot = (F + m_p * sin_theta * (l * theta_dot ** 2 + g * cos_theta)) / denominator
    theta_ddot = -(l * m_p * cos_theta * sin_theta * theta_dot ** 2 + F * cos_theta + (m_c + m_p) * g * sin_theta) / (l * denominator)

    return x_dot, x_ddot, theta_dot, theta_ddot

# Usage example:
pendulum = InvertedPendulum(mass_cart=1.0, mass_pendulum=0.1, length=1.0)
time = np.linspace(0, 10, 1000)
result = []
pendulum.add_force(0.01)

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


for i in range(len(time) - 1):
    if i == 100:
        pendulum.add_force(1.0)  # Modify this to the desired force value
    else:
        pendulum.add_force(0.0)

    segment_result = odeint(equations_of_motion, pendulum.get_state(), [time[i], time[i + 1]], args=(pendulum,))
    pendulum.state = segment_result[-1]
    result.append(segment_result[0])

result = np.vstack(result)
print(result)