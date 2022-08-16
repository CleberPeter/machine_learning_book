import numpy as np
import matplotlib.pyplot as plt

# [a b c] -> [1 a b c]
def add_ones_column(x):
    [m, n] = np.shape(x)
    new_x = np.zeros([m, n+1])
    new_x[:,0] = np.ones(m)
    new_x[:,1:] = x[:, 0:]

    return new_x

def sigma(z):
    return 1/(1 + np.exp(-z))

# logistic regression model
def h(x, o):
    x = add_ones_column(x)
    return sigma(np.dot(x, o))

# MSE/2
def half_MSE(x, o, y):
    m = np.shape(x)[0]
    h_x = h(x, o)
    return np.sum(np.power(h_x - y, 2), axis=0)*1/(2*m)

def log_without_nan(x):
    return np.nan_to_num(np.log(x))

# logistic regression cost
def logistic_regression_cost(x, o, y):
    m = np.shape(x)[0]
    h_x = h(x, o) # mxn
    y_t = np.transpose(y) # mx1 -> 1xm

    J = np.dot(y_t, (log_without_nan(h_x))) + np.dot((1-y_t),(log_without_nan(1 - h_x))) # 1xn

    return -J/m

def plot_cost(x, y, J, o):
    # prepare data in meshgrid format
    X, Y = np.meshgrid(o[0], o[1])
    
    # serialize meshgrid to use J
    c_o = np.power(len(o[1]), 2)
    o = np.zeros((2, c_o))
    o[0, :] = np.reshape(X, [1, c_o])
    o[1, :] = np.reshape(Y, [1, c_o])
    j = J(x, o, y)

    # transform j to meshgrid
    Z = np.reshape(j, np.shape(X))
    
    # plot
    fig = plt.figure(figsize=(20,10))
    # surface
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel(r'$\theta_{1}$')
    ax.set_ylabel(r'$\theta_{2}$')
    
    ax.plot_surface(X, Y, Z, alpha=0.2)
    
# plot sigma function
m = 100
x = np.linspace(-10, 10, m).reshape(m, 1)
y = sigma(x)

plt.plot(x, y)
plt.axvline(x = 0, color = 'k')
plt.axhline(y = 0.5, color = 'k')

# plot cost function
y = np.zeros(m).reshape(m, 1)
y[int(0.5*m):] = 1

# arbitrary data to visualize cost function
m = 100
o1 = np.linspace(-10, 10, m)
o2 = o1
plot_cost(x, y, half_MSE, [o1, o2])

# plot logistic regression cost function
x = np.linspace(1e-3, 999e-3, 100)
y_1 = -np.log(x)
y_0 = -np.log(1-x)
plt.figure()
plt.plot(x,y_1)
plt.plot(x,y_0)
plt.xlabel('h(x)')
plt.ylabel('J')
plt.legend(['y = 1', 'y = 0'])


o1 = np.linspace(-3, 3, m)
o2 = o1
plot_cost(x, y, logistic_regression_cost, [o1, o2])

plt.show()