import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
global anim 

def add_ones_column(x):
    """
    Append ones on first column ([a b c] -> [1 a b c])
    Arguments:
        x: np.array (mxn)
    Returns:
        np.array (mxn+1)
    """
    [m, n] = np.shape(x)
    new_x = np.zeros([m, n+1])
    new_x[:,0] = np.ones(m)
    new_x[:,1:] = x[:, 0:]

    return new_x

def h(x, o):
    """
    Linear regression model (o0*x0 + o1*x1)
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1xk)
    Returns:
        np.array (mxk)
    """
    x = add_ones_column(x)
    return np.dot(x, o)

def J(x, o, y):
    """
    Cost function -> half mean squared error
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1x1)
        y: np.array (mx1)
    Returns:
        double
    """
    m = np.shape(x)[0]
    h_x = h(x, o) # mx1
    e = h_x - y
    return np.dot(np.transpose(e), e)*1/(2*m)

def J_for_multiple_o_set(x, o, y):
    """
    Cost function for each o set
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1xk)
        y: np.array (mx1)
    Returns:
        np.array (1xk)
    """
    j_hist = []
    k = np.shape(o)[1]
    for i in range(k):
        o_k = np.vstack(o[:, i])
        j_hist.append(J(x, o_k, y))
    
    return np.hstack(j_hist)

def dJ(x, o, y):
    """
    Cost function gradient
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1x1)
        y: np.array (mx1)
    Returns:
        np.array (n+1x1) (cost gradient)
    """
    m = np.shape(x)[0]
    h_x = h(x, o) # mx1
    e = h_x - y
    x = add_ones_column(x)
    return np.dot(np.transpose(x), e)*1/m

def gradient_descent(x, o, y, alpha, min_grad, max_iterations):
    """
    Gradient descent -> Discover what o minimize j
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1x1)
        y: np.array (mx1)
        alpha: double (learning rate)
        min_grad: double (stop condition)
        max_iterations: int
    Returns:
        o: np.array (n+1x1)
        o_hist: np.array (n+1xi)
        i: int (iterations number)
    """
    i = 0
    o_hist = o
    grad = dJ(x, o, y)
    while np.linalg.norm(grad) > min_grad and i < max_iterations:
        o = o - alpha*grad
        o_hist = np.c_[o_hist, o] # append column

        grad = dJ(x, o, y)
        i += 1
        
    return [o, o_hist, i]

def momentum(x, o, y, alpha, gamma, min_grad, max_iterations):
    """
    Momentum gradient descent variation
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1x1)
        y: np.array (mx1)
        alpha: double (learning rate)
        gamma: double
        min_grad: double (stop condition)
        max_iterations: int
    Returns:
        o: np.array (n+1x1)
        o_hist: np.array (n+1xi)
        i: int (iterations number)
    """
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
    """
    RMSprop gradient descent variation
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1x1)
        y: np.array (mx1)
        alpha: double (learning rate)
        gamma: double
        eps: double
        min_grad: double (stop condition)
        max_iterations: int
    Returns:
        o: np.array (n+1x1)
        o_hist: np.array (n+1xi)
        i: int (iterations number)
    """
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
    """
    Adam gradient descent variation
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1x1)
        y: np.array (mx1)
        alpha: double (learning rate)
        beta_1: double
        beta_2: double
        eps: double
        min_grad: double (stop condition)
        max_iterations: int
    Returns:
        o: np.array (n+1x1)
        o_hist: np.array (n+1xi)
        i: int (iterations number)
    """
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

def plot_comparison_to_training(x, o, y):
    """
    Compare model with training data
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1x1)
        y: np.array (mx1)
    Returns:
    """
    h_x = h(x, o)
    
    plt.figure(figsize=(20,10))
    plt.plot(x, y, '*')
    plt.plot(x, h_x)
    plt.legend(['y', 'h_x = ' +  str(round(o[0,0],2)) + ' + ' + str(round(o[1,0],2)) + '*x'])

def plot_route(x, o_hist, y):
    """
    Plot cost curve in 3D surface, countour line and o_hist in these graphs
    Arguments:
        x: np.array (mxn)
        o_hist: np.array (n+1xk)
        y: np.array (mx1)
    Returns:
        
    """
    # arbitrary data to visualize cost function
    o0 = np.arange(-10, 10, 0.25)
    o1 = np.arange(-10, 10, 0.25)

    # prepare data in meshgrid format
    X, Y = np.meshgrid(o0, o1)
    
    # serialize meshgrid to use J
    c_o = np.power(len(o0), 2)
    o = np.zeros((2, c_o))
    o[0, :] = np.reshape(X, [1, c_o])
    o[1, :] = np.reshape(Y, [1, c_o])
    j = J_for_multiple_o_set(x, o, y)

    # transform j to meshgrid
    Z = np.reshape(j, np.shape(X))
    
    # get gradient descent route
    j = J_for_multiple_o_set(x, o_hist, y)
        
    # plot
    fig = plt.figure(figsize=(20,10))
    # surface
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel(r'$\theta_{0}$')
    ax.set_ylabel(r'$\theta_{1}$')
    
    ax.plot_surface(X, Y, Z, alpha=0.2)
    ax.plot(o_hist[0,:], o_hist[1,:], j[0], linestyle='--', marker='o')
    
    # plot contours    
    ax = fig.add_subplot(122)
    levels = [10, 50, 100, 250, 500, 750, 1000] # to improve view 
    contours = ax.contour(X, Y, Z, levels, colors='black')
    ax.plot(o_hist[0,:], o_hist[1,:], '*')

    plt.clabel(contours, inline = True, fontsize = 10)
    plt.xlabel(r'$\theta_{0}$')
    plt.ylabel(r'$\theta_{1}$')

def animate(j, line, o_hist):
    """
    Execute animation
    Arguments:
        j: int (iterations)
        line: ax.plot reference
        o_hist: np.array (n+1xk)
    Returns:
        
    """
    global anim 

    h_x = h(x, o_hist[:, j])

    line.set_data((x, h_x))
    plt.legend(['y', 'iter: ' + str(j) + ', equ: ' + str(round(o_hist[0,j],2)) + ' + (' + str(round(o_hist[1,j], 2)) + '*x)'])
    
    return line,
    
def show_animation(x, o_hist, y):
    """
    Show animation with of h(o) to y
    Arguments:
        x: np.array (mxn)
        o_hist: np.array (n+1xk)
        y: np.array (mx1)
    Returns:
        
    """
    global anim 
    
    fig, ax = plt.subplots(figsize=(12,6))
    line, = ax.plot([])     # A tuple unpacking to unpack the only plot

    plt.plot(x, y, '*', label='y')
    anim = FuncAnimation(fig, animate, repeat = False, frames=np.shape(o_hist)[1], blit=True, fargs=(line, o_hist), interval=1)
    plt.show()

# config gradient descent
max_iterations = 1
# alpha = 6e-2   # high learning rate
alpha = 1.5e-2   # good learning rate
# alpha = 1e-3   # low learning rate
min_grad = 1.0e-2

# define training data
x = np.mgrid[0:10:1].reshape(1,-1).T # mx1
y = 2 + 2*x 

# execute gradient descent to minimize o
o_start = [[10],[10]] # arbitrary start
[min_o, o_hist_gd, iterations] = gradient_descent(x, o_start, y, alpha, min_grad, max_iterations)

# plot result
print('iterations:', iterations)
plot_comparison_to_training(x, min_o , y)

plot_route(x, o_hist_gd, y)

# show_animation(x, o_hist_gd, y)

gamma = 0.9
[min_o, o_hist_momentum, iterations] = momentum(x, o_start, y, alpha, gamma, min_grad, max_iterations)
print('iterations:', iterations)
plot_route(x, o_hist_momentum, y)

gamma = 0.9
eps = 10**-8
[min_o, o_hist_RMSprop, iterations] = RMSprop(x, o_start, y, alpha, gamma, eps, min_grad, max_iterations)
print('iterations:', iterations)
plot_route(x, o_hist_RMSprop, y)

beta_1 = 0.7
beta_2 = 0.2
[min_o, o_hist_adam, iterations] = Adam(x, o_start, y, alpha, beta_1, beta_2, eps, min_grad, max_iterations)
print('iterations:', iterations)
plot_route(x, o_hist_adam, y)

j_gd       = J_for_multiple_o_set(x, o_hist_gd, y)
j_momentum = J_for_multiple_o_set(x, o_hist_momentum, y)
j_RMSprop  = J_for_multiple_o_set(x, o_hist_RMSprop, y)
j_adam     = J_for_multiple_o_set(x, o_hist_adam, y)
fig = plt.figure(figsize=(20,10))
plt.plot(j_gd[0])
plt.plot(j_momentum[0])
plt.plot(j_RMSprop[0])
plt.plot(j_adam[0])
plt.legend(['Gradient Descent', 'Momentum', 'RMSprop', 'Adam'])
plt.xlabel('iterations')
plt.ylabel(r'J($\theta)$')

plt.show()