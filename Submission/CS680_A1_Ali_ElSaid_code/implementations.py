# Python v3.10.0
# pandas==1.4.1
# numpy==1.21.4
# tqdm==4.64.1
# scipy==1.7.3

import numpy as np
import pandas as pd
from scipy import stats
from sympy import Q
from tqdm import tqdm
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------

########## E1.1 Perceptron Implementation
# Input:
# X: matrix with n rows and d columns representing the covariates, elements are real valued
# y: vector with n elements representing the response variable, elements are in {0,1}
# w: vector with d elements representing the initial w values, elements are real valued
# b: real number representing the initial b values
# max_pass: integer representing the number of epochs (passes on the whole data)
# Output:
# w: vector of size d, representing the final values of w after training 
# b: real number, representing the final value of b after training 
# mistake: vector of size max_pass, representing the number of mistakes per epoch
def perceptron(X, y, w=None, b=0, max_pass = 500):
    n, d = X.shape

    # Initialize w to zero vector if not given
    if w is None:
        w = np.zeros(d)
    else:
        w = w.copy()
    
    # Run algorithm
    mistake = np.zeros(max_pass)
    for t in tqdm(range(max_pass), "Epochs", leave=False):
        for i in tqdm(range(n), f"Epoch {t}", leave=False):
            xi = X[i, :]
            yi = y[i]
            if (yi*(np.dot(xi, w) + b) <= 0):
                w += yi*xi
                b += yi
                mistake[t] += 1
    return w, b, mistake
#-------------------------------------------------------------------------

########## E1.2 One-vs-All Multiclass Perceptron Implementation
# Input:
# X: matrix with d rows and n columns representing the covariates, elements are real valued
# y: vector with n elements representing the response variable, elements are in {1, 2, ..., c}, where c is the number of classes
# w: vector with d elements representing the initial w values for all binary perceptrons, elements are real valued
# b: real number representing the initial b values for all binary perceptrons
# max_pass: integer representing the number of epochs (passes on the whole data) for all binary perceptrons
# Output:
# W: matrix of size d * c, representing the final values of each w_i after training 
# b: vector of real numbers, representing the final values of each b_i after training 
# mistake: matrix of size max_pass * c, representing the number of mistakes per epoch in each class
def ova_perceptron(X, y, w_0=None, b_0=0, max_pass=500):
    # Initialize w_0 to zero vector if not given
    if w_0 is None:
        w_0 = np.zeros(X.shape[1])
    
    # List of classes
    classes = np.unique(y).astype(int)
    
    # Initialize W, b and mistake (storing values for each binary classifier)
    W = np.zeros((X.shape[1], classes.size))
    b = np.zeros(classes.size)
    mistake = np.zeros((max_pass, classes.size))
    
    for c in tqdm(classes, "Classes"):
        # Build new binary y (y = 1 if class = c, -1 otherwise)
        y_binary = 2*(y == c) - 1
        
        # Run binary perceptron
        w_c, b_c, mistake_c = perceptron(X, y_binary, w_0.copy(), b_0, max_pass)
        
        # Assign entries for classifier in dictionaries
        W[:,c-1] = w_c
        b[c-1] = b_c
        mistake[:,c-1] = mistake_c
    return W, b, mistake


# Function to test one vs all perceptron
def test_ova_perc(X, y, W, b):
    # Compute f(x) by doing the matrix operation (XW + 1b')
    #  where 1 is the 1 vector of size n_test, and b' is b transpose 
    fx = np.matmul(X, W) + np.outer(np.ones(X.shape[0]).T, b)

    # This will get us a matrix with size (n_test * # of classes), with each
    #  row being the f(x) under each binary classifier, so we just need to take
    #  argmax of each row to get the final prediction for each test point
    y_hat = np.argmax(fx, axis=1) + 1

    # Get no. of mistakes
    mistakes = np.sum(y != y_hat)
    
    return mistakes
#-------------------------------------------------------------------------

########## E1.3 One-vs-One Multiclass Perceptron Implementation
# Input:
# X: matrix with d rows and n columns representing the covariates, elements are real valued
# y: vector with n elements representing the response variable, elements are in {1, 2, ..., c}, where c is the number of classes
# w: vector with d elements representing the initial w values for all binary perceptrons, elements are real valued
# b: real number representing the initial b values for all binary perceptrons
# max_pass: integer representing the number of epochs (passes on the whole data) for all binary perceptrons
# Output:
# W: dictionary, representing the final values of each w_ij after training each binary perceptron 
# b: dictionary, representing the final values of each b_ij after training each binary perceptron 
# mistake: dictionary, representing the number of mistakes per epoch in each binary perceptron 
def ovo_perceptron(X, y, w_0=None, b_0=0, max_pass=500):
    # Initialize w_0 to zero vector if not given
    if w_0 is None:
        w_0 = np.zeros(X.shape[1])
    
    # List of classes
    classes = np.unique(y).astype(int)
    
    # Initialize W, b and mistake (storing values for each binary classifier)
    W = {}
    b = {}
    mistake = {}
    
    for i in range(1, classes.size+1):
        for j in range(i+1, classes.size+1):
            # Filter data on two classes, and transform to binary {1,-1}
            filt = (y == i) | (y == j)
            X_cur = X[filt]
            y_cur = 2*(y[filt] == i) - 1
        
            # Run binary perceptron
            w_ij, b_ij, mistake_ij = perceptron(X_cur, y_cur, w_0.copy(), b_0, max_pass)
        
            # Assign entries for classifier in dictionaries
            W[(i,j)] = w_ij
            b[(i,j)] = b_ij
            mistake[(i,j)] = mistake_ij
        
    return W, b, mistake


# Function to test one vs one perceptron
def test_ovo_perc(X, y, W, b):
    Y_hats = np.zeros((X.shape[0], len(W)))
    
    # Loop over all binary classifiers and compute their predictions 
    #  for all points
    k = 0
    for key, w in W.items():
        b_cur = b[key]
        # Compute f(x) by doing the matrix operation (Xw + 1b)
        #  where 1 is the 1 vector of size n_test,
        #  and b is the current b, and w is the current w vector
        fx = np.matmul(X, w) + b_cur * np.ones(X.shape[0])
        # Get the prediction y_hat by choosing the first class 
        #  if fx >= 0 and second class if fx < 0
        y_hat_cur = np.choose(fx < 0, [key[0], key[1]])
        Y_hats[:,k] = y_hat_cur
        k +=1
    
    # This will get us a matrix with size (n_test * # of binary predictors), with each
    #  row being the prediction under the different binary classifiers, so we just need to take
    #  mode of each row to get the final prediction for each test point
    y_hat = stats.mode(Y_hats, axis=1)[0].flatten()

    # Get no. of mistakes
    mistakes = np.sum(y != y_hat)
    
    return mistakes

#-------------------------------------------------------------------------

########## E1.6 Multiclass perceptron implementation
# Input:
# X: matrix with d rows and n columns representing the covariates, elements are real valued
# y: vector with n elements representing the response variable, elements are in {1, 2, ..., c}, where c is the number of classes
# W: matrix of size d * c, representing the initial w values, elements are real valued
# b: vector of real numbers, representing the initial b values 
# max_pass: integer representing the number of epochs (passes on the whole data)
# Output:
# W: matrix of size d * c, representing the final values of each w_k after training 
# b: vector of real numbers, representing the final values of each b_k after training 
# mistake: vector of size max_pass, representing the number of mistakes per epoch
def multiclass_perceptron(X, y, W=None, b=None, max_pass = 500):
    y = y.astype(int)
    n, d = X.shape
    classes = np.unique(y).astype(int)
    c = classes.size
    # Initialize W and b to zero if not given
    if W is None:
        W = np.zeros((d, c))
    
    if b is None:
        b = np.zeros(c)
    
    # Run algorithm
    mistake = np.zeros(max_pass)
    for t in tqdm(range(max_pass), "Epochs", leave=False):
        for i in tqdm(range(n), f"Epoch {t}", leave=False):
            xi = X[i, :]
            yi = y[i]
            y_hat = np.argmax(np.matmul(W.T, xi) + b) + 1
            if (yi != y_hat):
                W[:,yi-1] += xi
                b[yi-1] += 1
                W[:,y_hat-1] -= xi
                b[y_hat-1] -= 1
                mistake[t] += 1
    return W, b, mistake



#-------------------------------------------------------------------------
# Functions provided for data reading
def ReadX(path):
    print(f'>>> Reading data from: {path} ...')
    with open(path) as f:
        # only one line that includes everything
        file = f.readlines()

    print(f'#instances: {len(file)}') # 7352 for training set, 2947 for test set

    X_all = []
    for instance in file:
        f = filter(None, instance.split(' '))
        instance_filterd = list(f)
        instance_cleaned = [float(attr.strip()) for attr in instance_filterd]
        X_all.append(instance_cleaned)
    X_all = np.array(X_all)
    print('>>> Reading finished! Data are converted to numpy array.')
    print(f'shape of X: {X_all.shape} ==> each instance has {X_all.shape[1]} attributes.')

    return X_all

def ReadY(path):
    print(f'>>> Reading data from: {path} ...')
    with open(path) as f:
        # only one line that includes everything
        file = f.readlines()

        print(f'#instances: {len(file)}')  # 7352 for training set, 2947 for test set

    y_all = [float(label.strip()) for label in file]
    y_all = np.array(y_all)
    print('>>> Reading finished! Data are converted to numpy array.')
    print(f'shape of y: {y_all.shape}')
    return y_all
