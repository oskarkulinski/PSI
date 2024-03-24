import numpy as np
import matplotlib.pyplot as plt

f = lambda x,y: x**2 - y**2


def step_gradient_2d(x_current, y_current, learningRate):
    x_gradient = 2*x_current
    y_gradient = -2*y_current

    new_x = x_current - (learningRate * x_gradient)
    new_y = y_current - (learningRate * y_gradient)

    return [new_x, new_y]


def gradient_descent_runner_2d(starting_x, starting_y, learning_rate, num_iterations):
    x = starting_x
    y = starting_y
    x_history = [starting_x]
    y_history = [starting_y]
    for i in range(num_iterations):
        x, y = step_gradient_2d(x, y, learning_rate)
        x_history.append(x)
        y_history.append(y)
        #print(x, y)
    return x_history, y_history


learning_rate = 0.1
initial_x = 5 # initial y-intercept guess
initial_y = 1 # initial slope guess
num_iterations = 1000


x_history, y_history = gradient_descent_runner_2d(
    initial_x, initial_y, learning_rate, num_iterations)

x = np.arange(-10,10,0.02)
y = np.arange(-10,10,0.02)

X,Y= np.meshgrid(x,y)
Z = f(X,Y)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.plot(x_history, y_history, marker='o', color='red', label='Trajektoria')
plt.scatter(x_history[0], y_history[0], color='blue', label='Punkt startowy')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_history, y_history, f(np.array(x_history), np.array(y_history)), marker='o', color='red', label='Trajektoria')
ax.scatter(x_history[0], y_history[0], f(x_history[0], y_history[0]), color='blue', label='Punkt startowy')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
plt.show()


###############################
# 3.5
###############################

learning_rate = 0.1
initial_x = 5 # initial y-intercept guess
initial_y = 0 # initial slope guess
num_iterations = 1000


x_history, y_history = gradient_descent_runner_2d(
    initial_x, initial_y, learning_rate, num_iterations)

x = np.arange(-10,10,0.02)
y = np.arange(-10,10,0.02)

X,Y= np.meshgrid(x,y)
Z = f(X,Y)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.plot(x_history, y_history, marker='o', color='red', label='Trajektoria')
plt.scatter(x_history[0], y_history[0], color='blue', label='Punkt startowy')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_history, y_history, f(np.array(x_history), np.array(y_history)), marker='o', color='red', label='Trajektoria')
ax.scatter(x_history[0], y_history[0], f(x_history[0], y_history[0]), color='blue', label='Punkt startowy')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
plt.show()