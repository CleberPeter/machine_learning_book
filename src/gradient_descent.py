import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# linear regression hypothesis: o1 + o2*x
def h(x, o):
    m = len(x)
    input = np.zeros((m, 2))
    input[:,0] = np.ones(m)
    input[:,1] = x[:, 0]
    return np.dot(input, o)

# J = MSE/2
def J(x, o, y):
    m = len(x)
    h_x = h(x, o)
    return np.sum(np.power(h_x - y, 2), axis=0)*1/(2*m)

def dJ(x, o, y):
    m = len(x)
    h_x = h(x, o)
    dJ1 = np.sum(h_x - y)*1/m
    dJ2 = np.sum((h_x - y)*x)*1/m
    return np.array([[dJ1],[dJ2]])

def gradient_descent(x, o, y, alpha, min_grad, max_iterations):
    i = 0
    o_hist = o
    grad = dJ(x, o, y)
    while np.linalg.norm(grad) > min_grad and i < max_iterations:
        o = o - alpha*grad
        o_hist = np.c_[o_hist, o] # append column

        grad = dJ(x, o, y)
        i += 1
        
    return [o, o_hist, i]

def plot_comparison_to_training(x, min_o, y):
    h_x = h(x, min_o)
    
    plt.figure(figsize=(20,10))
    plt.plot(x, y, '*')
    plt.plot(x, h_x)
    plt.legend(['y', 'h_x = ' +  str(round(min_o[0,0],2)) + ' + ' + str(round(min_o[1,0],2)) + '*x'])

def plot_route(x, o_hist, y):
    # arbitrary data to visualize cost function
    o1 = np.arange(-10, 10, 0.25)
    o2 = np.arange(-10, 10, 0.25)

    # prepare data in meshgrid format
    X, Y = np.meshgrid(o1, o2)
    
    # serialize meshgrid to use J
    c_o = np.power(len(o1), 2)
    o = np.zeros((2, c_o))
    o[0, :] = np.reshape(X, [1, c_o])
    o[1, :] = np.reshape(Y, [1, c_o])
    j = J(x, o, y)

    # transform j to meshgrid
    Z = np.reshape(j, np.shape(X))
    
    # get gradient descent route
    j = J(x, o_hist, y)
        
    # plot
    fig = plt.figure(figsize=(20,10))
    # surface
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel(r'$\theta_{1}$')
    ax.set_ylabel(r'$\theta_{2}$')
    
    ax.plot_surface(X, Y, Z, alpha=0.2)
    ax.plot(o_hist[0,:], o_hist[1,:], j, linestyle='--', marker='o')
    
    # plot contours    
    ax = fig.add_subplot(122)
    levels = [10, 50, 100, 250, 500, 750, 1000] # to improve view 
    contours = ax.contour(X, Y, Z, levels, colors='black')
    ax.plot(o_hist[0,:], o_hist[1,:], '*')

    plt.clabel(contours, inline = True, fontsize = 10)
    plt.xlabel(r'$\theta_{1}$')
    plt.ylabel(r'$\theta_{2}$')

global anim 
def animate(j, line, o_hist):
    global anim 

    h_x = h(x, o_hist[:, j])

    line.set_data((x, h_x))
    plt.legend(['y', 'iter: ' + str(j) + ', equ: ' + str(round(o_hist[0,j],2)) + ' + (' + str(round(o_hist[1,j], 2)) + '*x)'])
    
    return line,
    
def show_animation(x, o_hist, y):
    global anim 
    
    fig, ax = plt.subplots(figsize=(12,6))
    line, = ax.plot([])     # A tuple unpacking to unpack the only plot

    plt.plot(x, y, '*', label='y')
    anim = FuncAnimation(fig, animate, repeat = False, frames=np.shape(o_hist)[1], blit=True, fargs=(line, o_hist), interval=1)
    plt.show()

def momentum(x, o, y, alpha, gamma, min_grad, max_iterations):
    i = 0
    o_hist = o
    grad = dJ(x, o, y)
    v = 0
    while np.linalg.norm(grad) > min_grad and i < max_iterations:
        v = gamma*v + alpha*grad
        o = o - v
        o_hist = np.c_[o_hist, o] # append column

        grad = dJ(x, o, y)
        i += 1
        
    return [o, o_hist, i]

def RMSprop(x, o, y, alpha, gamma, eps, min_grad, max_iterations):
    i = 0
    o_hist = o
    grad = dJ(x, o, y)
    v = 0
    while np.linalg.norm(grad) > min_grad and i < max_iterations:
        v = gamma*v + (1 - gamma)*grad**2
        o = o - alpha*grad/np.sqrt(v + eps)
        o_hist = np.c_[o_hist, o] # append column

        grad = dJ(x, o, y)
        i += 1
        
    return [o, o_hist, i]

def Adam(x, o, y, alpha, beta_1, beta_2, eps, min_grad, max_iterations):
    i = 0
    o_hist = o
    grad = dJ(x, o, y)
    v = m = 0
    while np.linalg.norm(grad) > min_grad and i < max_iterations:
        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * (grad**2)
        mhat = m / (1 - beta_1)
        vhat = v / (1 - beta_2)
        o = o - alpha*mhat/np.sqrt(vhat + eps)
        o_hist = np.c_[o_hist, o] # append column

        grad = dJ(x, o, y)
        i += 1
        
    return [o, o_hist, i]

# config gradient descent
max_iterations = 10000
# alpha = 6e-2   # high learning rate
alpha = 1.5e-2   # good learning rate
# alpha = 1e-3   # low learning rate
min_grad = 1.0e-2

# define training data
x = np.arange(0, 10, 1)[:, np.newaxis] # column vector
m = len(x)
y = 2 + 2*x 

# execute gradient descent to minimize o
o_start = [[10],[10]] # arbitrary start
[min_o, o_hist_gd, iterations] = gradient_descent(x, o_start, y, alpha, min_grad, max_iterations)

# plot result
print('iterations:', iterations)
plot_comparison_to_training(x, min_o , y)

# plot route
plot_route(x, o_hist_gd, y)

# show animation
show_animation(x, o_hist_gd, y)

# momentum
gamma = 0.9
[min_o, o_hist_momentum, iterations] = momentum(x, o_start, y, alpha, gamma, min_grad, max_iterations)
print('iterations:', iterations)
plot_route(x, o_hist_momentum, y)

# RMSprop
gamma = 0.9
eps = 10**-8
[min_o, o_hist_RMSprop, iterations] = RMSprop(x, o_start, y, alpha, gamma, eps, min_grad, max_iterations)
print('iterations:', iterations)
plot_route(x, o_hist_RMSprop, y)

# Adam
beta_1 = 0.7
beta_2 = 0.2
[min_o, o_hist_adam, iterations] = Adam(x, o_start, y, alpha, beta_1, beta_2, eps, min_grad, max_iterations)
print('iterations:', iterations)
plot_route(x, o_hist_adam, y)

#comparison
j_gd       = J(x, o_hist_gd, y)
j_momentum = J(x, o_hist_momentum, y)
j_RMSprop  = J(x, o_hist_RMSprop, y)
j_adam     = J(x, o_hist_adam, y)
fig = plt.figure(figsize=(20,10))
plt.plot(j_gd)
plt.plot(j_momentum)
plt.plot(j_RMSprop)
plt.plot(j_adam)
plt.legend(['Gradient Descent', 'Momentum', 'RMSprop', 'Adam'])

plt.show()