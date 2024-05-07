import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x ** 2

x = np.linspace(-5, 5, 200)
y = f(x)
plt.plot(x, y, '--k', ms=10);


def step_gradient_1d(x_current, learningRate):
    x_gradient = 2 * x_current
    new_x = x_current - learningRate * x_gradient

    plt.arrow(x_current, f(x_current), - (learningRate * x_gradient), -(f(x_current) - f(new_x)),
              head_width=0.05, head_length=0.5, ec="red")

    return new_x


def gradient_descent_runner_1d(starting_x, learning_rate, num_iterations):
    x = starting_x
    print(x)
    for i in range(num_iterations):
        x = step_gradient_1d(x, learning_rate)
        #print(x)
    return x


learning_rate = 0.2
initial_x = 5
num_iterations = 30
x = gradient_descent_runner_1d(initial_x, 0.001, num_iterations)
plt.show()

x = gradient_descent_runner_1d(initial_x, 0.1, num_iterations)
plt.show()

x = gradient_descent_runner_1d(initial_x, 0.2, num_iterations)
plt.show()

x = gradient_descent_runner_1d(initial_x, 0.5, num_iterations)
plt.show()

x = gradient_descent_runner_1d(initial_x, 0.9, num_iterations)
plt.show()

x = gradient_descent_runner_1d(initial_x, 0.99, num_iterations)
plt.show()

x = gradient_descent_runner_1d(initial_x, 0.999, num_iterations)
plt.show()



###########

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
#%matplotlib notebook

plt.close('all')

fun = lambda x,y: 4*x**2+y**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data.
X = np.arange(-7, 7, 0.25)
Y = np.arange(-7, 7, 0.25)
X, Y = np.meshgrid(X, Y)
Z = fun(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0.01, antialiased=True, alpha=0.3)


#####################################

def step_gradient_2d(x_current, y_current, learningRate):
    x_gradient = 8 * x_current - 2
    y_gradient = 2*y_current

    new_x = x_current - x_gradient * learning_rate
    new_y = y_current - y_gradient * learning_rate

    ax.quiver(x_current, y_current, (fun(x_current, y_current)) ,
              - (learningRate * x_gradient), - (learningRate * y_gradient),
              (-(fun(x_current,y_current)-fun(new_x,new_y))))

    return [new_x, new_y]

def gradient_descent_runner_2d(starting_x, starting_y, learning_rate, num_iterations):
    x = starting_x
    y = starting_y
    for i in range(num_iterations):
        x, y = step_gradient_2d(x, y, learning_rate)
        #print(x, y)
    return [x, y]


learning_rate = 0.9
initial_x = 0 # initial y-intercept guess
initial_y = 5 # initial slope guess
num_iterations = 10
[x, y] = gradient_descent_runner_2d(initial_x, initial_y, learning_rate, num_iterations)
plt.plot([initial_x],[initial_y],[fun(initial_x,initial_y)],"ok")
plt.show()

#####################################

learning_rate = 0.9
initial_x = 5 # initial y-intercept guess
initial_y = 0 # initial slope guess
num_iterations = 10
[x, y] = gradient_descent_runner_2d(initial_x, initial_y, learning_rate, num_iterations)
plt.plot([initial_x],[initial_y],[fun(initial_x,initial_y)],"ok")
plt.show()

### 2 ###

import numpy as np
import matplotlib.pyplot as plt

chi2 = lambda x,y: 4*x**2-2*x+y**2

x = np.arange(-10,10,0.02)
y = np.arange(-10,10,0.02)

X,Y= np.meshgrid(x,y)

Z = chi2(X,Y)

plt.figure()
CS = plt.contour(X,Y,Z)

plt.plot([5],[5],"o")

#####################################

def step_gradient_2d(x_current, y_current, learningRate):
    x_gradient = 8*x_current-2
    y_gradient = 2*y_current

    new_x = x_current - (learningRate * x_gradient)
    new_y = y_current - (learningRate * y_gradient)

    plt.arrow(x_current, y_current, - (learningRate * x_gradient), - (learningRate * y_gradient), head_width=0.05, head_length=0.5,ec="red")

    return [new_x, new_y]
def gradient_descent_runner_2d(starting_x, starting_y, learning_rate, num_iterations):
    x = starting_x
    y = starting_y
    for i in range(num_iterations):
        x, y = step_gradient_2d(x, y, learning_rate)
        #print(x, y)
    return [x, y]


learning_rate = 0.3
initial_x = 0 # initial y-intercept guess
initial_y = 5 # initial slope guess
num_iterations = 1000
[x, y] = gradient_descent_runner_2d(initial_x, initial_y, learning_rate, num_iterations)


#####################################
plt.axis('equal')
plt.show()



### 3 ###


import numpy as np
import matplotlib.pyplot as plt

chi2 = lambda x,y: 4*x**2-2*x+y**2

x = np.arange(-10,10,0.02)
y = np.arange(-10,10,0.02)

X,Y= np.meshgrid(x,y)

Z = chi2(X,Y)

plt.figure()
CS = plt.contour(X,Y,Z)

plt.plot([5],[5],"o")

#####################################

def step_gradient_2d(x_current, y_current, learningRate):
    x_gradient = 2 * x_current
    y_gradient = -2 * y_current

    new_x = x_current - (learningRate * x_gradient)
    new_y = y_current - (learningRate * y_gradient)

    plt.arrow(x_current, y_current, - (learningRate * x_gradient), - (learningRate * y_gradient), head_width=0.05, head_length=0.5,ec="red")

    return [new_x, new_y]
def gradient_descent_runner_2d(starting_x, starting_y, learning_rate, num_iterations):
    x = starting_x
    y = starting_y
    for i in range(num_iterations):
        x, y = step_gradient_2d(x, y, learning_rate)
        #print(x, y)
    return [x, y]


learning_rate = 0.3
initial_x = 5 # initial y-intercept guess
initial_y = 1 # initial slope guess
num_iterations = 1000
[x, y] = gradient_descent_runner_2d(initial_x, initial_y, learning_rate, num_iterations)

#####################################
plt.axis('equal')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Make data.
X = np.arange(-7, 7, 0.25)
Y = np.arange(-7, 7, 0.25)
X, Y = np.meshgrid(X, Y)
Z = fun(X, Y)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0.01, antialiased=True, alpha=0.3)
plt.plot(x,y)
plt.show()


plt.figure()
CS = plt.contour(X,Y,Z)

plt.plot([5],[5],"o")


learning_rate = 0.3
initial_x = 5 # initial y-intercept guess
initial_y = 0 # initial slope guess
num_iterations = 1000
[x, y] = gradient_descent_runner_2d(initial_x, initial_y, learning_rate, num_iterations)

#####################################
plt.axis('equal')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Make data.
X = np.arange(-7, 7, 0.25)
Y = np.arange(-7, 7, 0.25)
X, Y = np.meshgrid(X, Y)
Z = fun(X, Y)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0.01, antialiased=True, alpha=0.3)
plt.plot(x,y)
plt.show()