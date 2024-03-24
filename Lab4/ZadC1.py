import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2

x = np.linspace(-5, 5, 200)
y = f(x)
plt.plot(x, y, '--k', ms=10)


def step_gradient_1d(x_current, learningRate):
    x_gradient = 2*x_current
    new_x = x_current - learningRate * x_gradient

    plt.arrow(x_current, f(x_current), - (learningRate * x_gradient), -(f(x_current)-f(new_x)),
              head_width=0.05, head_length=0.5,ec="red")

    return new_x


def gradient_descent_runner_1d(starting_x, learning_rate, num_iterations):
    x = starting_x
    print(x)
    for i in range(num_iterations):
        x = step_gradient_1d(x,learning_rate)
        print(x)
    return x

learning_rate = 0.2
initial_x = 5
num_iterations = 30
x = gradient_descent_runner_1d(initial_x, learning_rate, num_iterations)

plt.show()



lr1 = 0.001
x = gradient_descent_runner_1d(initial_x, lr1, num_iterations)
print(x, end='\n\n')
lr2 = 0.1
x = gradient_descent_runner_1d(initial_x, lr2, num_iterations)
print(x, end='\n\n')
lr3 = 0.2
x = gradient_descent_runner_1d(initial_x, lr3, num_iterations)
print(x, end='\n\n')
lr4 = 0.5
x = gradient_descent_runner_1d(initial_x, lr4, num_iterations)
print(x, end='\n\n')
lr5 = 0.9
x = gradient_descent_runner_1d(initial_x, lr5, num_iterations)
print(x, end='\n\n')
lr6 = 0.99
x = gradient_descent_runner_1d(initial_x, lr6, num_iterations)
print(x, end='\n\n')
lr7 = 0.999
x = gradient_descent_runner_1d(initial_x, lr7, num_iterations)
print(x, end='\n\n')

