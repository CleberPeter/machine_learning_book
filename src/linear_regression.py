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
    Linear regression model (o0*x0 + o1*x1)
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1xk)
    Returns:
        np.array (mxk)
    """
    x = add_ones_column(x)
    return np.dot(x, o)

def MSE(x, o, y):
    """
    Mean squared error
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1xk)
        y: np.array (mx1)
    Returns:
        double (mse)
    """
    m = np.shape(x)[0]
    h_x = h(x, o)
    e = h_x - y
    return np.dot(np.transpose(e), e)*1/m

def MAE(x, o, y):
    """
    Mean absolute error
    Arguments:
        x: np.array (mxn) 
        o: np.array (n+1xk)
        y: np.array (mx1)
    Returns:
        double (mae)
    """
    m = np.shape(x)[0]
    h_x = h(x, o)
    return np.sum(np.abs(h_x - y), axis=0)*1/m

def minimize_j(J, x, o, y):
    """
    Discover which one o minimize j
    Arguments:
        J: cost function
        x: np.array (mxn)
        o: np.array (n+1xk)
        y: np.array (mx1)
    Returns:
        o_min: np.array (n+1x1) (o minimized)
    """
    j_min = np.inf
    o_min = []
    k = np.shape(o)[1]
    for i in range(k):
        o_k = np.vstack(o[:, i])
        j = J(x, o_k, y)

        if j < j_min:
            j_min = j
            o_min = o_k
            
    return o_min

def compare_mse_and_mae(x, o, y):
    """
    Compares MSE and MAE ploting your curves minimized
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1xk)
        y: np.array (mx1)
    Returns:
        
    """
    o_min_mse = minimize_j(MSE, x, o ,y)
    h_mse = h(x, o_min_mse)
    
    o_min_mae = minimize_j(MAE, x, o ,y)
    h_mae = h(x, o_min_mae)

    plt.figure(figsize=(20,8))

    plt.plot(x, y, '*')
    plt.plot(x, h_mse)
    plt.plot(x, h_mae)
    
    h_mse_legend = 'h_mse = ' + str(np.round(o_min_mse[0],2)) + ' + ' + str(np.round(o_min_mse[1],2)) + '*x'
    h_mae_legend = 'h_mae = ' + str(np.round(o_min_mae[0],2)) + ' + ' + str(np.round(o_min_mae[1],2)) + '*x'

    plt.legend(['y', h_mse_legend, h_mae_legend])

x = np.mgrid[-10:10:1].reshape(1,-1).T # mx1
y = 1 + 2*x

k = 10000
o = np.random.normal(size = [2, k]) # 2xn_o

compare_mse_and_mae(x, o, y)

# introduce outlier
y[3] *= 10

compare_mse_and_mae(x, o, y)

# arbitrary parameters
o0 = np.arange(-10, 10, 0.25)
o1 = np.arange(-10, 10, 0.25)

# prepare data in meshgrid format
X, Y = np.meshgrid(o0, o1)

# serialize meshgrid to use MAE
c_o = np.power(len(o0), 2)
o = np.zeros((2, c_o))
o[0, :] = np.reshape(X, [1, c_o]) 
o[1, :] = np.reshape(Y, [1, c_o])
mae = MAE(x, o, y)

# transform mae to meshgrid
Z = np.reshape(mae, np.shape(X))

# plot
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel(r'$\theta_{0}$')
ax.set_ylabel(r'$\theta_{1}$')
ax.plot_surface(X, Y, Z)

# plot countour lines
fig = plt.figure(figsize=(20,10))
contours = plt.contour(X, Y, Z, 20)
plt.plot(1,2, '*') # center point -> minimum cost
plt.clabel(contours, inline = True, fontsize = 10)
plt.xlabel(r'$\theta_{0}$')
plt.ylabel(r'$\theta_{1}$')

plt.show()