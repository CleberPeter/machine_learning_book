import matplotlib.pyplot as plt
import numpy as np

# multiple linear regression hypothesis: o1*x1 + o2*x2
def h(x, o):
    return np.dot(x, o) # mx2 . 2x1 -> mx1

# J = MSE/2
def J(x, o, y):
    m = np.shape(x)[0]
    h_x = h(x, o)
    return np.sum(np.power(h_x - y, 2), axis=0)*1/(2*m)

def dJ(x, o, y):
    m = np.shape(x)[0]
    h_x = h(x, o)
    x = np.transpose(x) # mx2 -> 2xm
    e = h_x - y # mx1
    dJ = np.dot(x, e)*1/m # 2xm . mx1 = 2x1
    return dJ

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

# model o
o1 = 1
o2 = 1
o = np.array([[o1],[o2]])
# execute gradient descent to minimize o
o_start = [[10],[10]] # arbitrary start
# config gradient descent
max_iterations = 10000
# alpha = 6e-2   # high learning rate
alpha = 1.5e-2   # good learning rate
# alpha = 1e-3   # low learning rate
min_grad = 1.0e-2

# same range 
x = np.mgrid[-10:10:1, -10:10:1].reshape(2,-1).T # mx2
y = h(x,o)
[min_o, o_hist_gd, iterations] = gradient_descent(x, o_start, y, alpha, min_grad, max_iterations)
print('iterations:', iterations)
plot_route(x, o_hist_gd, y)

# different ranges
alpha = 2e-3   # changed learning rate to prevent divergence
x = np.mgrid[-10:10:1, -30:30:3].reshape(2,-1).T # mx2
y = h(x,o)
[min_o, o_hist_gd, iterations] = gradient_descent(x, o_start, y, alpha, min_grad, max_iterations)
print('iterations:', iterations)
plot_route(x, o_hist_gd, y)


plt.show()