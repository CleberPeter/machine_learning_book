import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def accuracy(h_x, y):
    """
    Accuracy from classification
    Arguments:
        h_x: np.array (mx1)
        y: np.array (mx1)
    Returns:
        double
    """
    m = np.shape(h_x)[0]
    hints = 0
    for h_x_i, y_i in zip(h_x, y):
        if h_x_i == y_i:
            hints += 1
    return hints/m

def binarize(x):
    """
    Continuous to discrete binary mapping (x >= 0.5 ? 1 : 0)
    Arguments:
        x: np.array (mx1)
    Returns:
        np.array (mx1)
    """
    return np.heaviside(x-0.5, 1)

def boundary_decision(x, o):
    """
    Boundary decision for h = o0 + o1x1 + o2x2
    Arguments:
        x: np.array (mxn)
        o: np.array (n+1x1)
    Returns:
        np.array (mx1)
    """
    return -o[1]/o[2]*x[:,0] -o[0]/o[2] 

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

# import dataset
df = pd.read_csv('datasets/classifier_binary/dogs_cats.csv')
df.head()

# preprocess the output
plt.figure(figsize=(20,10))

df['classe'].replace(['cachorro', 'gato'], [0, 1], inplace=True)
df_cats = df.loc[df['classe'] == 1]
df_dogs = df.loc[df['classe'] == 0]

plt.scatter(df_cats['comprimento'], df_cats['peso'])
plt.scatter(df_dogs['comprimento'], df_dogs['peso'])
plt.xlabel('comprimento')
plt.ylabel('peso')
plt.legend(['gato', 'cachorro'])

x = np.vstack(df[['comprimento', 'peso']].values) # mx2
y = np.vstack(df['classe'].values) # mx1

# config optimizer
o_start = np.array([[0],[0],[0]]) # arbitrary start
# config gradient descent
max_iterations = 10000
alpha = 1e-2
min_grad = 1e-1

[min_o, o_hist_gd, iterations] = gradient_descent(x, o_start, y, alpha, min_grad, max_iterations)
print('min_o:', min_o)
print('iterations:', iterations)

# plot mapping result
h_x_image = np.linspace(0, 1, 1000)
plt.figure(figsize=(20,10))
plt.plot(h_x_image, binarize(h_x_image))

# get model accuracy
h_x = h(x, min_o)
h_x = binarize(h_x)
print('accuracy: ', accuracy(h_x, y))

# show classification process
plt.figure(figsize=(20,10))

plt.scatter(df_cats['comprimento'], df_cats['peso'])
plt.scatter(df_dogs['comprimento'], df_dogs['peso'])

bd = boundary_decision(x, min_o)
plt.plot(x[:, 0], bd)

plt.xlabel('comprimento')
plt.ylabel('peso')
plt.legend(['gato', 'cachorro', 'decision boundary'])

plt.show()