###########

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

# %matplotlib notebook

plt.close('all')

fun = lambda x, y: 4 * x ** 2 + y ** 2

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-7, 7, 0.25)
Y = np.arange(-7, 7, 0.25)
X, Y = np.meshgrid(X, Y)
Z = fun(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm',
                       linewidth=0.01, antialiased=True, alpha=0.3)


#####################################

def step_gradient_2d(x_current, y_current, learningRate):
    x_gradient = 8 * x_current - 2
    y_gradient = 2 * y_current

    new_x = x_current - learningRate * x_gradient
    new_y = y_current - learningRate * y_gradient

    ax.quiver(x_current, y_current, (fun(x_current, y_current)),
              - (learningRate * x_gradient), - (learningRate * y_gradient),
              (-(fun(x_current, y_current) - fun(new_x, new_y))))

    return [new_x, new_y]


def gradient_descent_runner_2d(starting_x, starting_y, learning_rate, num_iterations):
    x = starting_x
    y = starting_y
    for i in range(num_iterations):
        x, y = step_gradient_2d(x, y, learning_rate)
        # print(x, y)
    return [x, y]


learning_rate = 0.9
initial_x = 0  # initial y-intercept guess
initial_y = 5  # initial slope guess
num_iterations = 10
[x, y] = gradient_descent_runner_2d(initial_x, initial_y, learning_rate, num_iterations)

#####################################

plt.plot([initial_x], [initial_y], [fun(initial_x, initial_y)], "ok")
plt.show()

#####################################
plt.close('all')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm',
                       linewidth=0.01, antialiased=True, alpha=0.3)

learning_rate = 0.9
initial_x = 5  # initial y-intercept guess
initial_y = 0  # initial slope guess
num_iterations = 10
[x, y] = gradient_descent_runner_2d(
    initial_x, initial_y, learning_rate, num_iterations)

plt.plot([initial_x], [initial_y], [fun(initial_x, initial_y)], "ok")
plt.show()
############################
# Zadanie 2.5
############################


chi2 = lambda x, y: x ** 2 - 2 * x + y ** 2

x = np.arange(-10, 10, 0.02)
y = np.arange(-10, 10, 0.02)

X, Y = np.meshgrid(x, y)

Z = chi2(X, Y)

plt.figure()
CS = plt.contour(X, Y, Z)

plt.plot([5], [5], "o")


#####################################

def step_gradient_2d_f2(x_current, y_current, learningRate):
    x_gradient = 2 * x_current - 2
    y_gradient = 2 * y_current

    new_x = x_current - (learningRate * x_gradient)
    new_y = y_current - (learningRate * y_gradient)

    plt.arrow(x_current, y_current, - (learningRate * x_gradient), - (learningRate * y_gradient), head_width=0.05,
              head_length=0.5, ec="red")

    return [new_x, new_y]


def gradient_descent_runner_2d_f2(starting_x, starting_y, learning_rate, num_iterations):
    x = starting_x
    y = starting_y
    for i in range(num_iterations):
        x, y = step_gradient_2d_f2(x, y, learning_rate)
        # print(x, y)
    return [x, y]


learning_rate = 0.1
initial_x = 5  # initial y-intercept guess
initial_y = 5  # initial slope guess
num_iterations = 1000
[x, y] = gradient_descent_runner_2d_f2(initial_x, initial_y, learning_rate, num_iterations)

#####################################
plt.axis('equal')
plt.show()


####################################

plt.figure()
CS = plt.contour(X,Y,Z)
plt.plot([5],[5],"o")

learning_rate = 0.1
initial_x = 5  # initial y-intercept guess
initial_y = 5  # initial slope guess
num_iterations = 1000
[x, y] = gradient_descent_runner_2d(initial_x, initial_y, learning_rate, num_iterations)

#####################################
plt.axis('equal')
plt.show()