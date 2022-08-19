import numpy as np
import matplotlib.pyplot as plt

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
    Multiple linear regression hypothesis: o0*x0 + o1*x1 + o2*x2 + on*xn
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
    o1 = np.arange(-10, 10, 0.25)
    o2 = np.arange(-10, 10, 0.25)

    # prepare data in meshgrid format
    X, Y = np.meshgrid(o1, o2)
    
    # serialize meshgrid to use J
    c_o = np.power(len(o1), 2)
    o = np.zeros((3, c_o))
    o[1, :] = np.reshape(X, [1, c_o])
    o[2, :] = np.reshape(Y, [1, c_o])
    j = J_for_multiple_o_set(x, o, y)

    # transform j to meshgrid
    Z = np.reshape(j, np.shape(X))
    
    # get gradient descent route
    j = J_for_multiple_o_set(x, o_hist, y)
        
    # plot
    fig = plt.figure(figsize=(20,10))
    # surface
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel(r'$\theta_{1}$')
    ax.set_ylabel(r'$\theta_{2}$')
    
    ax.plot_surface(X, Y, Z, alpha=0.2)
    ax.plot(o_hist[1,:], o_hist[2,:], j[0], linestyle='--', marker='o')
    
    # plot contours    
    ax = fig.add_subplot(122)
    contours = ax.contour(X, Y, Z, colors='black')
    ax.plot(o_hist[1,:], o_hist[2,:], '*')

    plt.clabel(contours, inline = True, fontsize = 10)
    plt.xlabel(r'$\theta_{1}$')
    plt.ylabel(r'$\theta_{2}$')

def norm(x):
    """
    Nomalize x
    Arguments:
        x: np.array (mxn)
    Returns:
        np.array (mxn)
    """
    # axis = 0 -> by column, 1 -> by row
    u = np.mean(x, axis = 0)
    s = np.ptp(x, axis = 0)
    return (x - u)/s

def normal_equation(x, y):
    """
    Normal equation to find o wich minimize J
    Arguments:
        x: np.array (mxn)
        y: np.array (mx1)
    Returns:
        np.array (n+1x1) (o)
    """
    x = add_ones_column(x)
    x_t = np.transpose(x)
    return np.dot(np.dot(np.linalg.inv(np.dot(x_t, x)), x_t), y)

# model o
o1 = 0
o2 = 1
o3 = 1
o = np.array([[o1],[o2],[o3]])
# execute gradient descent to minimize o
o_start = [[0],[10],[10]] # arbitrary start
# config gradient descent
max_iterations = 10000
# alpha = 1.5e-2   # good learning rate
alpha = 2e-3   # caution with learning rate to prevent divergence
min_grad = 1.0e-2

x = np.mgrid[-10:10:1, -30:30:3].reshape(2,-1).T # mx2
y = h(x,o)
[min_o, o_hist_gd, iterations] = gradient_descent(x, o_start, y, alpha, min_grad, max_iterations)
print(min_o)
print('iterations:', iterations)
plot_route(x, o_hist_gd, y)

alpha = 3   # good learning rate
x_norm = norm(x)
y = h(x_norm,o)
[min_o, o_hist_gd, iterations] = gradient_descent(x_norm, o_start, y, alpha, min_grad, max_iterations)
print(min_o)
print('iterations:', iterations)
plot_route(x_norm, o_hist_gd, y)

#analitycal_solution
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
fig = plt.figure(figsize=(20,10))
plt.plot(x, y)

# find o
alpha = 1   # good learning rate
o_start = np.array([[10],[10],[10],[10]]) # arbitrary start
x_norm = norm(x_cubic)
[min_o, o_hist_gd, iterations] = gradient_descent(x_norm, o_start, y, alpha, min_grad, max_iterations)
print(min_o)
print(iterations)

j_hist = J_for_multiple_o_set(x_norm, o_hist_gd, y)
y_min = h(x_norm, min_o)
fig = plt.figure(figsize=(20,10))

# cost evolution
ax = fig.add_subplot(121)
plt.plot(j_hist[0])
# comparison
print('MSE:', J(x_norm, min_o, y))
ax = fig.add_subplot(122)
plt.plot(x, y_min)
plt.plot(x, y)
plt.legend(['y_min', 'y'])

plt.show()