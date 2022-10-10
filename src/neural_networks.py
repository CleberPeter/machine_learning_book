import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndimage
from scipy.optimize import minimize

SEED = 17
np.random.seed(SEED)
    
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
    x = add_ones_column(x) # bias
    a_l = x
    a = [a_l]
    for o in o_list:
        a_l = sigmoid(np.dot(a_l, o))
        a_l = add_ones_column(a_l) # bias
        a.append(a_l)
    
    a[-1] = a[-1][:, 1:] # remove bias from output
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

def logistic_regression_cost(o_vect, x, y, layers_sizes):
    """
    Logistic regression cost
    Arguments:
        o_vect: np.array (kx1)
        x: np.array (mxn)
        y: np.array (mxn)
        hidden_layers_sizes: list of np.array (rxs)
    Returns:
        double (half mse)
    """

    o_list = vect_to_list(o_vect, layers_sizes)
    h_x = forward_propagation(x, o_list)[-1]
    m = np.shape(x)[0]
    J = 0
    for h_x_i, y_i in zip(h_x, y):
        J += np.dot(-y_i, log_without_nan(h_x_i)) - np.dot((1-y_i), log_without_nan(1 - h_x_i))

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
    
def backpropagation(o_vect, x, y, layers_sizes):
    """
    Backpropagation
    Arguments:
        o_vect: np.array (kx1)
        x: np.array (mxn)
        y: np.array (mxn)
        hidden_layers_sizes: list of np.array (rxs)
    Returns:
        list of np.array (kx1).
    """

    o_list = vect_to_list(o_vect, layers_sizes)
    m = np.shape(x)[0]
    a = forward_propagation(x, o_list)
    L = len(a) - 1

    a_L = a[L] # output from model
    delta_l = a_L - y # (-1/m) * (y/a_L - np.nan_to_num((1 - y)/(1-a_L))) * d_sigmoid(a_L)
    
    # backwards
    dJ_o = []
    l = L - 1
    for o in reversed(o_list):
        dJ_o_i = np.dot(np.transpose(a[l]), delta_l)
        # dJ_o_i[0, :] = 0 # don't update bias
        dJ_o.append(dJ_o_i)
        
        delta_l = np.dot(delta_l, np.transpose(o)) * d_sigmoid(a[l]) 
        delta_l = delta_l[:, 1:] # remove ignore bias
        l = l - 1
    
    dJ_o_list = list(reversed(dJ_o)) # reorder
    return list_to_vect(dJ_o_list)

def gradient_descent(o_vect, x, y, layers_sizes, alpha, max_iterations):
    """
    Gradient descent -> Discover what o minimize j
    Arguments:
        x: np.array (mxn)
        o_list: list of np.array (rxs). 
                First should be r = n+1 and last s = k
        y: np.array (mxk)
        alpha: double (learning rate)
        max_iterations: int
    Returns:
        o_list: list of np.array (rxs)
        i: int (iterations number)
        j_hist: list of double
    """
    i = 0
    grad_vect = backpropagation(o_vect, x, y, layers_sizes)
    grad = vect_to_list(grad_vect, layers_sizes)
    o_list = vect_to_list(o_vect, layers_sizes)
    j_hist = []
    while i < max_iterations:
        for idx, row in enumerate(o_list):
            o_list[idx] = o_list[idx] - alpha*grad[idx]
        
        o_vect = list_to_vect(o_list)
        j_hist.append(logistic_regression_cost(o_vect, x, y, layers_sizes))
        
        grad_vect = backpropagation(o_vect, x, y, layers_sizes)
        grad = vect_to_list(grad_vect, layers_sizes)
        
        i += 1

    return [o_list, i, j_hist]

def init_o(layers_sizes):
    """
    Ramdonly initialize the parameters model 
    Arguments:
        layers_sizes: list of integers
    Returns:
        np.array (kx1)
    """
    o_layers = []
    last_layer_size = layers_sizes[0]
    for layer_size in layers_sizes[1:]:
        epsilon = np.sqrt(6) / np.sqrt(last_layer_size + layer_size)
        w = (np.random.rand(layer_size, last_layer_size + 1) * 2 * epsilon) - epsilon
        o_layers.append(w)

        last_layer_size = layer_size

    return list_to_vect(o_layers)

def list_to_vect(x):
    """
    Transform matrix list to 1-D vector 
    Arguments:
        x: list of np.array (rxs)
    Returns:
        np.array (kx1)
    """
    vect = x[0].flatten()
    for x_i in x[1:]:
        vect = np.concatenate((vect, x_i), axis=None)
    
    return vect

def vect_to_list(x, layers_sizes):
    """
    Transform 1-D vector to matrix list according layers_sizes  
    Arguments:
        x: np.array (kx1)
    Returns:
        list of np.array (rxs)
    """
    layer_data_list = []
    last_layer_size = layers_sizes[0]
    last_layer_idx_data = 0
    for layer_size in layers_sizes[1:]:
        layer_vector_size = (last_layer_size + 1) * layer_size
        layer_vector_data = x[last_layer_idx_data : last_layer_idx_data+layer_vector_size]
        layer_data = layer_vector_data.reshape((last_layer_size + 1, layer_size))
        
        layer_data_list.append(layer_data)

        last_layer_idx_data += layer_vector_size
        last_layer_size = layer_size
    
    return layer_data_list

def show_image(X, n_images, size_image = [20, 20]):
    """
    Show image in gray scale
    Arguments:
        X: np.array (mxn)
        n_images: int
        size_image: np.array (rxs) -> r.s = n
    Returns:
    """
    
    n_images_sqrt = int(np.sqrt(n_images))
    f, axarr = plt.subplots(n_images_sqrt, n_images_sqrt)
    
    image = 0
    for i in range(n_images_sqrt):
        for j in range(n_images_sqrt):
            x = X[image, :].reshape(size_image)
            x_rot = ndimage.rotate(x, 90, reshape=True)

            axarr[i, j].imshow(x_rot, cmap='gray', origin='lower')
            axarr[i, j].axis('off')
            image += 1

def vectorize(x, n_labels):
    """
    Transform integer into array like: 2 -> [0 0 1 0 ... 0]
    Arguments:
        x: np.array (mx1)
        n_labels: int
    Returns:
        np.array (mxn_labels)
    """
    m = np.shape(x)[0]
    X = np.zeros([m, n_labels])

    for i, x_i in enumerate(x):
        X[i, x_i] = 1

    return X

def monitoring_minimize(o_vect):
    global iterations

    J = logistic_regression_cost(o_vect, X, Y, layers_sizes)
    print('iterations:', iterations, '| cost: ', J)
    
    j_hist.append(J)
    iterations += 1

    return False

def predict(x):
    """
    Multiclass prediction
    Arguments:
        x: np.array (mxn)
    Returns:
        np.array (mx1)
    """
    [m, n] = np.shape(x)
    preds = np.zeros((m,n))
    for idx, x_i in enumerate(x):
        pred = np.zeros((1,n))
        out = np.argmax(x_i, axis=0)
        pred[0, out] = 1
        preds[idx, :] = pred

    return preds

def accuracy(h_x, y):
    """
    Accuracy from classification
    Arguments:
        h_x: np.array (mx1)
        y: np.array (mx1)
    Returns:
        accuracy: double
        id_error_list: list of integer
    """
    m = np.shape(h_x)[0]
    hints = 0
    id_error_list = []
    for idx, (h_x_i, y_i) in enumerate(zip(h_x, y)):
        if np.array_equal(h_x_i, y_i):
            hints += 1
        else:
            id_error_list.append(idx)

    return [hints/m, id_error_list]

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
max_iterations = 1000
hidden_layers_sizes = [2] # [8]

layers_sizes = [np.shape(x)[1]]
layers_sizes.extend(hidden_layers_sizes)
layers_sizes.append(np.shape(y)[1])

o_vect = init_o(layers_sizes)
[min_o, i, j_hist] = gradient_descent(o_vect, x, y, layers_sizes, alpha, max_iterations)
print('iterations:', i)
print(forward_propagation(x, min_o)[-1])

plt.figure(figsize=(20,10))
plt.plot(j_hist)

# handwritten digits

df = pd.read_csv('datasets/classifier_multiclass/handwritten_digits.csv', header=None)
X = df.iloc[:, 0:400].values
m = np.shape(X)[0]
y = df.iloc[:, 400]

print('shape: ', df.shape)
print('classes:')
print(y.value_counts())
n_images = 64
indexes = np.random.randint(0, m, n_images)
show_image(X[indexes], n_images)

hidden_layers_sizes = [25]
n_labels = 10
Y = vectorize(y, n_labels)

layers_sizes = [np.shape(X)[1]]
layers_sizes.extend(hidden_layers_sizes)
layers_sizes.append(np.shape(Y)[1])

o_vect = init_o(layers_sizes)
max_iterations = 50 # 100
iterations = 0
j_hist = []
print('starting minimize ...')
result = minimize(logistic_regression_cost, o_vect, args=(X, Y, layers_sizes), jac=backpropagation, method = 'CG', options = {'maxiter' : max_iterations}, callback=monitoring_minimize)
o_min_vect = result.x

plt.figure()
plt.plot(j_hist)

o_min_list = vect_to_list(o_min_vect, layers_sizes)
a = forward_propagation(X, o_min_list)
pred = predict(a[-1])

[accy, id_error_list] = accuracy(pred, Y)
print('accuracy:', accy)

errors = len(id_error_list)
if errors:
    show_image(X[id_error_list], errors)

plt.show()