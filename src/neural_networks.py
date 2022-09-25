import numpy as np
import matplotlib.pyplot as plt

SEED = 17

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

def sigmoid(z):
    """
    Logistic function
    Arguments:
        z: np.array (mx1)
    Returns:
        np.array (mx1)
    """
    return 1/(1 + np.exp(-z))

def forward_propagation(x, o_list):
    """
    Forward Propagation
    Arguments:
        x: np.array (mxn)
        o_list: list of np.array (rxs). 
                First should be r = n+1 and last s = k 
    Returns:
        list of np.array (mxs)
    """
    a = [x]
    a_l = add_ones_column(x) # bias
    for o in o_list:
        a_l = sigmoid(np.dot(a_l, o))
        a.append(a_l)
        a_l = add_ones_column(a_l) # bias

    return a

def log_without_nan(x):
    """
    Transforms nan to number
    Arguments:
        x: np.array (mx1)
    Returns:
        np.array (mx1)
    """
    return np.nan_to_num(np.log(x))

def logistic_regression_cost(h_x, y):
    """
    Logistic regression cost
    Arguments:
        h_x: np.array (mx1)
        y: np.array (mx1)
    Returns:
        double (half mse)
    """
    m = np.shape(y)[0]
    y_t = np.transpose(y) # mx1 -> 1xm

    J = np.dot(-y_t, (log_without_nan(h_x))) - np.dot((1-y_t),(log_without_nan(1 - h_x))) # 1xn

    return J/m

def d_sigmoid(g_z):
    """
    Derivate of sigmoid function
    Arguments:
        g_z: np.array (mx1)
    Returns:
        np.array (mx1)
    """
    return g_z * (1 - g_z) 
    
def backpropagation(x, o_list, y):
    """
    Backpropagation
    Arguments:
        x: np.array (mxn)
        o_list: list of np.array (rxs). 
                First should be r = n+1 and last s = k
        y: np.array (mxk)
    Returns:
        list of np.array (rxs).
    """
    m = np.shape(x)[0]
    a = forward_propagation(x, o_list)
    L = len(a) - 1

    a_L = a[L] # output from model
    delta_l = (-1/m) * (y/a_L - (1 - y)/(1-a_L)) * d_sigmoid(a_L)
    
    # backwards
    dJ_o = []
    l = L - 1
    for o in reversed(o_list):
        dJ_o.append(np.dot(np.transpose(a[l]), delta_l))
        delta_l = np.dot(delta_l, np.transpose(o[1:])) * d_sigmoid(a[l]) # o[1:] to ignore bias
        l = l - 1
        
    return list(reversed(dJ_o)) # reorder

def gradient_descent(x, o_list, y, alpha, max_iterations, min_error):
    """
    Gradient descent -> Discover what o minimize j
    Arguments:
        x: np.array (mxn)
        o_list: list of np.array (rxs). 
                First should be r = n+1 and last s = k
        y: np.array (mxk)
        alpha: double (learning rate)
        max_iterations: int
        min_error: double (stop condition)
    Returns:
        o_list: list of np.array (rxs)
        i: int (iterations number)
        j_hist: list of double
    """
    i = 0
    grad = backpropagation(x, o_list, y)
    j_hist = []
    while np.linalg.norm(grad[-1]) > min_error and i < max_iterations:
        for o_l, grad_l in zip(o_list, grad):
            o_l[1:] = o_l[1:] - alpha*grad_l
        
        h_x = forward_propagation(x, o_list)[-1]
        j_hist.append(logistic_regression_cost(h_x, y)[0])
        
        grad = backpropagation(x, o_list, y)
        i += 1

    return [o_list, i, j_hist]

def neural_network_classifier(x, y, hidden_layers_sizes, alpha, max_iterations, min_error):
    """
    Neural network classifier
    Arguments:
        x: np.array (mxn)
        y: np.array (mxk)
        hidden_layers_sizes: list of np.array (rxs)
        alpha: double (learning rate)
        max_iterations: int
        min_error: double (stop condition)
    Returns:
        o_list: list of np.array (rxs)
        i: int (iterations number)
        j_hist: list of double
    """
    
    o_start_list = []
    last_hidden_layer_size = np.shape(x)[1]
    np.random.seed(SEED)
    for hidden_layer_size in hidden_layers_sizes:
        o_start_list.append(np.random.rand(last_hidden_layer_size + 1, hidden_layer_size))
        last_hidden_layer_size = hidden_layer_size
    
    o_start_list.append(np.random.rand(last_hidden_layer_size + 1, np.shape(y)[1]))
    return gradient_descent(x, o_start_list, y, alpha, max_iterations, min_error)

# XOR operator
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 4x2

o_list = []
o_1 = np.array([[-30, 10], [20, -20], [20, -20]]) # 3x2
o_list.append(o_1)
o_2 = np.array([[-10], [20], [20]]) # 3x1
o_list.append(o_2)

print(forward_propagation(x, o_list)[-1])

# XOR expected output
y = np.array([[1], [0], [0], [1]]) # 4x1

alpha = 1e-1
max_iterations = 30000 # 7500
min_error = 1e-4
hidden_layers_sizes = (2,) # (8,)
[min_o, i, j_hist] = neural_network_classifier(x, y, hidden_layers_sizes, alpha, max_iterations, min_error)
print('iterations:', i)
print(forward_propagation(x, min_o)[-1])

plt.figure(figsize=(20,10))
plt.plot(j_hist)
plt.show()
