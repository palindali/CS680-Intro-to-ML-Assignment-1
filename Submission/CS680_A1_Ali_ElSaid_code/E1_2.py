# Exercise 1, Question 2
# (implementation of one vs all multiclass perceptron in implementations.py)

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from implementations import ova_perceptron, test_ova_perc, ReadX, ReadY

X_trn = ReadX("activity_X_train.txt")
y_trn = ReadY("activity_y_train.txt")
X_tst = ReadX("activity_X_test.txt")
y_tst = ReadY("activity_y_test.txt")
W, b, mistake = ova_perceptron(X_trn, y_trn, max_pass=500)

trn_errors = test_ova_perc(X_trn, y_trn, W, b)
print(f'Training errors: {trn_errors}')
tst_errors = test_ova_perc(X_tst, y_tst, W, b)
print(f'Testing errors: {tst_errors}')

# Training errors: 139
# Testing errors: 145
