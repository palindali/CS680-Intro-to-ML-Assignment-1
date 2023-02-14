# Exercise 1, Question 1
# (implementation of perceptron in implementations.py)

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from implementations import perceptron

# Read data
X = pd.read_csv("spambase_X.csv", header=None).to_numpy().T
y = pd.read_csv("spambase_y.csv", header=None).iloc[:,0].to_numpy()
# Initialize w and b to zeros
w_0 = np.zeros(X.shape[1])
b_0 = 0
max_pass = 500
w, b, mistake = perceptron(X, y, w_0, b_0, max_pass)

plt.plot(range(max_pass), mistake)
plt.xlabel("# of passes")
plt.ylabel("# of mistakes")
plt.title("E1.1: Mistakes per epoch for perceptron on spambase")
plt.show()
