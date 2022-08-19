import numpy as np
import matplotlib.pyplot as plt

def sigma(z):
    """
    Logistic function
    Arguments:
        z: np.array (mx1)
    Returns:
        np.array (mx1)
    """
    return 1/(1 + np.exp(-z))

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
    Logistic regression model: sigma(o^Tx)
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1xk)
    Returns:
        np.array (mxk)
    """
    x = add_ones_column(x)
    return sigma(np.dot(x, o))

def half_MSE(x, o, y):
    """
    Half mean squared error
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1xk)
        y: np.array (mx1)
    Returns:
        double (half mse)
    """
    m = np.shape(x)[0]
    h_x = h(x, o)
    e = h_x - y
    return np.dot(np.transpose(e), e)*1/(2*m)

def J_for_multiple_o_set(x, o, y, J):
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

def plot_cost(x, o, y, J):
    """
    Plot J
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1xk)
        y: np.array (mx1)
        J: point of function
    Returns:

    """
    # prepare data in meshgrid format
    X, Y = np.meshgrid(o[0], o[1])
    
    # serialize meshgrid to use J
    c_o = np.power(len(o[1]), 2)
    o = np.zeros((2, c_o))
    o[0, :] = np.reshape(X, [1, c_o])
    o[1, :] = np.reshape(Y, [1, c_o])
    j = J_for_multiple_o_set(x, o, y, J)

    # transform j to meshgrid
    Z = np.reshape(j, np.shape(X))
    
    # plot
    fig = plt.figure(figsize=(20,10))
    # surface
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel(r'$\theta_{0}$')
    ax.set_ylabel(r'$\theta_{1}$')
    
    ax.plot_surface(X, Y, Z, alpha=0.2)

def log_without_nan(x):
    """
    Transforms nan to number
    Arguments:
        x: np.array (mx1)
    Returns:
        np.array (mx1)
    """
    return np.nan_to_num(np.log(x))

def logistic_regression_cost(x, o, y):
    """
    Logistic regression cost
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1xk)
        y: np.array (mx1)
    Returns:
        double (half mse)
    """
    m = np.shape(x)[0]
    h_x = h(x, o) # mxn
    y_t = np.transpose(y) # mx1 -> 1xm

    J = np.dot(y_t, (log_without_nan(h_x))) + np.dot((1-y_t),(log_without_nan(1 - h_x))) # 1xn

    return -J/m

# plot sigma function
m = 100
x = np.linspace(-10, 10, m).reshape(m, 1)
y = sigma(x)

fig = plt.figure(figsize=(20,10))
plt.plot(x, y)
plt.axvline(x = 0, color = 'k')
plt.axhline(y = 0.5, color = 'k')

# arbitrary data to visualize cost function
y = np.zeros(m).reshape(m, 1)
y[int(0.5*m):] = 1
m = 100
o1 = np.linspace(-10, 10, m)
o2 = o1
plot_cost(x, [o1, o2], y, half_MSE)

# plot log cost
x = np.linspace(1e-3, 999e-3, 1000)
y_1 = -np.log(x)
y_0 = -np.log(1-x)
fig = plt.figure(figsize=(20,10))
plt.plot(x,y_1)
plt.plot(x,y_0)
plt.xlabel('h(x)')
plt.ylabel('J')
plt.legend(['y = 1', 'y = 0'])

# plot logistic regression cost
x = np.linspace(-10, 10, m).reshape(m, 1)
o1 = np.linspace(-3, 3, m)
o2 = o1
plot_cost(x, [o1, o2], y, logistic_regression_cost)

plt.show()