import matplotlib.pyplot as plt
import numpy as np

# [a b c] -> [1 a b c]
def add_ones_column(x):
    [m, n] = np.shape(x)
    new_x = np.zeros([m, n+1])
    new_x[:,0] = np.ones(m)
    new_x[:,1:] = x[:, 0:]

    return new_x

# multiple linear regression hypothesis: o1 + o2*x1 + o3*x2 + on+1*xn
def h(x, o):
    x = add_ones_column(x)
    return np.dot(x, o) # mx(n+1) . (n+1)x1 = mx1

def MSE(h_x, y):
    m = len(h_x)
    return np.sum(np.power(h_x - y, 2), axis=0)*1/m

# J = MSE/2
def J(x, o, y):
    m = np.shape(x)[0]
    h_x = h(x, o)
    return np.sum(np.power(h_x - y, 2), axis=0)*1/(2*m)

def dJ(x, o, y):
    [m, n] = np.shape(x)
    h_x = h(x, o)
    e = h_x - y # mx1

    x = np.transpose(x) # mxn -> nxm
    
    dJ = np.zeros([n+1, 1])   
    dJ[0] = np.sum(e)/m
    dJ[1:] = np.dot(x, e)/m # nxm . mx1 -> nx1
    return dJ # n+1x1

def plot_route(x, o_hist, y):
    # arbitrary data to visualize cost function
    o2 = np.arange(-10, 10, 0.25)
    o3 = np.arange(-10, 10, 0.25)

    # prepare data in meshgrid format
    X, Y = np.meshgrid(o2, o3)
    
    # serialize meshgrid to use J
    c_o = np.power(len(o2), 2)
    o = np.zeros((3, c_o))
    o[1, :] = np.reshape(X, [1, c_o])
    o[2, :] = np.reshape(Y, [1, c_o])
    j = J(x, o, y)

    # transform j to meshgrid
    Z = np.reshape(j, np.shape(X))
    
    # get gradient descent route
    j = J(x, o_hist, y)
        
    # plot
    fig = plt.figure(figsize=(20,10))
    # surface
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel(r'$\theta_{2}$')
    ax.set_ylabel(r'$\theta_{3}$')
    
    ax.plot_surface(X, Y, Z, alpha=0.2)
    ax.plot(o_hist[1,:], o_hist[2,:], j, linestyle='--', marker='o')
    
    # plot contours    
    ax = fig.add_subplot(122)
    contours = ax.contour(X, Y, Z, colors='black')
    ax.plot(o_hist[1,:], o_hist[2,:], '*')

    plt.clabel(contours, inline = True, fontsize = 10)
    plt.xlabel(r'$\theta_{2}$')
    plt.ylabel(r'$\theta_{3}$')

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

def norm(x): # axis = 0 -> by column, 1 -> by row
    u = np.mean(x, axis = 0)
    s = np.ptp(x, axis = 0)
    return (x - u)/s

def normal_equation(x, y):
    x = add_ones_column(x)
    x_t = np.transpose(x)
    return np.dot(np.dot(np.linalg.inv(np.dot(x_t, x)), x_t), y)

# model o
o1 = 0
o2 = 1
o3 = 1
o = np.array([[o1],[o2],[o3]])
# execute gradient descent to minimize o
o_start = np.array([[0],[10],[10]]) # arbitrary start
# config gradient descent
max_iterations = 10000
# alpha = 1.5e-2   # good learning rate
alpha = 2e-3   # caution with learning rate to prevent divergence
min_grad = 1.0e-2

x = np.mgrid[-10:10:1, -30:30:3].reshape(2,-1).T # mx2
y = h(x,o)
[min_o, o_hist_gd, iterations] = gradient_descent(x, o_start, y, alpha, min_grad, max_iterations)
print('min_o:', min_o)
print('iterations:', iterations)
plot_route(x, o_hist_gd, y)

# inputs normalized
alpha = 3   # good learning rate
x_norm = norm(x)
y = h(x_norm,o)
[min_o, o_hist_gd, iterations] = gradient_descent(x_norm, o_start, y, alpha, min_grad, max_iterations)
print('min_o:', min_o)
print('iterations:', iterations)
plot_route(x, o_hist_gd, y)

# analitycal_solution
x = np.mgrid[-10:10:1, -30:30:3].reshape(2,-1).T # mx2
y = h(x,o)
print('min_o:', normal_equation(x, y)) 

# polynomial regression
x = np.mgrid[-10:10:1].reshape(1,-1).T # mx1
o = np.array([[-8], [-2], [8], [2]])
x1 = x
x2 = np.power(x, 2)
x3 = np.power(x, 3)
x_cubic = np.column_stack((x1, x2, x3)) # mx3
y = h(x_cubic, o)
plt.figure()
plt.plot(x, y)

# find o
alpha = 1   # good learning rate
o_start = np.array([[10],[10],[10],[10]]) # arbitrary start
x_norm = norm(x_cubic)
[min_o, o_hist_gd, iterations] = gradient_descent(x_norm, o_start, y, alpha, min_grad, max_iterations)
print(min_o)
print(iterations)

j_hist = J(x_norm, o_hist_gd, y)
y_min = h(x_norm, min_o)
print('MSE:', MSE(y_min, y))
plt.plot(x, y_min)

plt.figure()
plt.plot(j_hist)

plt.show()
