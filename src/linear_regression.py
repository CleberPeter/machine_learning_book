import matplotlib.pyplot as plt
import numpy as np

# linear regression hypothesis: o1 + o2*x
def h(x, o):
    m = len(x)
    input = np.zeros([m, 2])
    input[:,0] = np.ones(m)
    input[:,1] = x[:, 0]
    
    return np.dot(input, o)

def MSE(x, o, y):
    m = len(x)
    h_x = h(x, o)
    return np.sum(np.power(h_x - y, 2), axis=0)*1/m

def MAE(x, o, y):
    m = len(x)
    h_x = h(x, o)
    return np.sum(np.abs(h_x - y), axis=0)*1/m

def minimize(f, x, o, y):
    f_x = f(x, o, y)
    f_x_min_idx = np.argmin(f_x)
    f_x_min = f_x[f_x_min_idx]
    f_x_min_o = o[:, f_x_min_idx]
    h_min_o = h(x, f_x_min_o)
    return [f_x_min_o, f_x_min, h_min_o]

def compute(x, o, y):
    # MSE like cost function
    [min_o_mse, mse_min, h_mse] = minimize(MSE, x, o ,y)

    # MAE like cost function
    [min_o_mae, mae_min, h_mae] = minimize(MAE, x, o ,y)

    plt.figure(1)

    plt.plot(x, y, '*')
    plt.plot(x, h_mse)
    plt.plot(x, h_mae)
    
    h_mse_legend = 'h_mse = ' + str(round(min_o_mse[0],2)) + ' + ' + str(round(min_o_mse[1],2)) + '*x'
    h_mae_legend = 'h_mae = ' + str(round(min_o_mae[0],2)) + ' + ' + str(round(min_o_mae[1],2)) + '*x'

    plt.legend(['y', h_mse_legend, h_mae_legend])
    
x = np.arange(0, 10, 1)[:, np.newaxis] # column vector
m = len(x)
y = 1 + 2*x

# random [[o1],[o2]] parameter values
n_o = 10000
o = np.random.normal(size = [2, n_o])

compute(x, o, y)

# introduce outlier
y[3] *= 10

compute(x, o, y)

## VISUALIZATION METHODS ##

# arbitrary parameters 
o1 = np.arange(-10, 10, 0.25)
o2 = np.arange(-10, 10, 0.25)

# prepare data in meshgrid format
X, Y = np.meshgrid(o1, o2)
# serialize meshgrid to use MAE
c_o = np.power(len(o1),2)
o = np.zeros((2, c_o))
o[0, :] = np.reshape(X, [1, c_o]) 
o[1, :] = np.reshape(Y, [1, c_o])
mae = MAE(x, o, y)
# transform mae to meshgrid
Z = np.reshape(mae, np.shape(X))

#plot
fig = plt.figure(2)
ax = fig.add_subplot(projection='3d')
ax.set_xlabel(r'$\theta_{1}$')
ax.set_ylabel(r'$\theta_{2}$')
ax.plot_surface(X, Y, Z)

# plot countour lines
contours = plt.contour(X, Y, Z, 20)
plt.plot(1,2, '*') # center point -> minimum cost
plt.clabel(contours, inline = True, fontsize = 10)
plt.xlabel(r'$\theta_{1}$')
plt.ylabel(r'$\theta_{2}$')

plt.show()