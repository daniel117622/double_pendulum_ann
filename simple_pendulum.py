from scipy.integrate import odeint
import numpy as np
import random
from sklearn.linear_model import LinearRegression

class SimplePendulum:  # Equivalent class for Simple Pendulum
    def __init__(self, length, gravity=9.81):
        self.length = length
        self.gravity = gravity
        self.state = np.array([0.0, 0.0, 0.0])  # theta, theta_dot, theta_ddot

    def set_state(self, theta, theta_dot, theta_ddot):
        self.state = np.array([theta, theta_dot, theta_ddot])

    def get_state(self):
        return self.state

    
def equations_of_motion(state, t, simple_pendulum):
    l = simple_pendulum.length
    g = simple_pendulum.gravity

    theta, theta_dot , theta_ddot = state

    theta_ddot = -g / l * np.sin(theta)
    
    return theta, theta_dot, theta_ddot

pendulum = SimplePendulum(length=1.0)
time = np.linspace(0, 10, 1000)
result = []


float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
V1 = [] ; V2 = []
pendulum.set_state(np.pi / 9, 0 , 0) 
for i in range(len(time) - 1):
    if i % 20 == 0:
        theta      = pendulum.get_state()[0]
        theta_dot  = pendulum.get_state()[1]
        theta_ddot = pendulum.get_state()[2]
        V1.append([np.sin(theta) , np.cos(theta), theta]) # Append the sine of the angle
        V2.append(theta_ddot)

    segment_result = odeint(equations_of_motion, pendulum.get_state(), [time[i], time[i + 1]], args=(pendulum,))
    pendulum.state = segment_result[-1]
    result.append(segment_result[0])

V1 = np.array(V1)
V2 = np.array(V2)

model = LinearRegression().fit(V1,V2)
print(model.coef_)
print(model.intercept_)