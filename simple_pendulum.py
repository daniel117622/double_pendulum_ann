from scipy.integrate import odeint
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

class SimplePendulum:
    def __init__(self, length, gravity=9.81):
        self.length = length
        self.gravity = gravity
        self.state = np.array([0.0, 0.0])  # theta, theta_dot

    def set_state(self, theta, theta_dot):
        self.state = np.array([theta, theta_dot])

    def get_state(self):
        return self.state


def equations_of_motion(state, t, simple_pendulum):
    l = simple_pendulum.length
    g = simple_pendulum.gravity
    theta, theta_dot = state
    theta_ddot = -g / l * np.sin(theta)
    return theta_dot, theta_ddot


pendulum = SimplePendulum(length=1.0)
time = np.linspace(0, 10, 1000)
result = []

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
V1 = []
V2 = []

pendulum.set_state(np.pi / 4, 0)  # Initial state

theta_ddot_values = []
for i in range(len(time) - 1):
    segment_result = odeint(equations_of_motion, pendulum.get_state(), [time[i], time[i + 1]], args=(pendulum,))
    pendulum.set_state(*segment_result[-1])
    theta, theta_dot = pendulum.get_state()
    _, theta_ddot = equations_of_motion(pendulum.get_state(), time[i], pendulum)
    theta_ddot_values.append(theta_ddot)
    V1.append([theta, theta_dot])

V1 = np.array(V1)
V2 = np.array(V2)
theta_values = V1[:, 0]
theta_ddot_values = np.array(theta_ddot_values)

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Define the candidate models
def model1(theta, a): return a * theta
def model2(theta_dot, a): return a * theta_dot
def model3(theta_dot, a): return a * theta_dot**2
def model4(theta, a): return a * np.cos(theta)
def model5(theta, a): return a * np.sin(theta)

# Separate the variables
theta_values = V1[:-1, 0]
theta_dot_values = V1[:-1, 1]

# List of models to try
models = [model1, model2, model3, model4, model5]
names = ["Linear theta", "Linear theta_dot", "Quadratic theta_dot", "Cosine theta", "Sine theta"]

# Loop over models
for name, model in zip(names, models):
    # Select the appropriate independent variable
    x_values = theta_values if model in [model1, model4, model5] else theta_dot_values
    # Curve fitting
    params, _ = curve_fit(model, x_values, theta_ddot_values[0:len(theta_ddot_values)-1])
    # Compute fitted values
    fitted_values = model(x_values, *params)
    # Compute R^2 score
    r2 = r2_score(theta_ddot_values[0:len(theta_ddot_values)-1], fitted_values)
    print(f"Model: {name}, R^2 score: {r2}")