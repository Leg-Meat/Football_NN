import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This program is an incredibly basic Neural Network designed on estimating what
# the result of a football match SHOULD have been given the statistics


test_data = pd.read_csv('Data/Test/E3_202627.csv')

train_data_1 = pd.read_csv('Data/Train/E3_202526.csv')
train_data_2 = pd.read_csv('Data/Train/E3_202425.csv')
train_data_3 = pd.read_csv('Data/Train/E3_202324.csv')
train_data = pd.concat([train_data_1, train_data_2, train_data_3], axis=0, ignore_index=True)

# relevant Data columns
relevant_cols = ['Date', 'Time',
                 'HomeTeam', 'AwayTeam',
                 'FTHG', 'FTAG', 'HTHG', 'HTAG',
                 'HS', 'AS', 'HST', 'AST', 'HF', 'AF',
                 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']

train_data = np.array(train_data[relevant_cols])
test_data = np.array(test_data[relevant_cols])

m, n = train_data.shape
np.random.shuffle(train_data)

# Test data splits
info_test = test_data[:, 0:4]
label_test = test_data[:, 4:8].T
feature_test = test_data[:, 8:].T

# Dev data splits
dev_data = train_data[0:50]
info_dev = dev_data[:, 0:4].T
label_dev = dev_data[:, 4:8].T
feature_dev = dev_data[:, 8:].T

# Training data splits
train_data = train_data[50:]
info_train = train_data[:, 0:4].T
label_train = train_data[:, 4:8].T.astype(float)
feature_train = train_data[:, 8:].T.astype(float)

# Normalise features
feature_train = (feature_train - feature_train.min(axis=1, keepdims=True)) / (
    feature_train.max(axis=1, keepdims=True) - feature_train.min(axis=1, keepdims=True) + 1e-8
)

feature_test = (feature_test - feature_test.min(axis=1, keepdims=True)) / (
    feature_test.max(axis=1, keepdims=True) - feature_test.min(axis=1, keepdims=True) + 1e-8
)

feature_dev = (feature_dev - feature_dev.min(axis=1, keepdims=True)) / (
    feature_dev.max(axis=1, keepdims=True) - feature_test.min(axis=1, keepdims=True) + 1e-8
)

def init_parameters():
    w1 = np.random.rand(4, 12)
    b1 = np.random.rand(4, 1)
    w2 = np.random.rand(4, 4)
    b2 = np.random.rand(4, 1)

    return w1, b1, w2, b2

def ReLU(x):
    return np.maximum(0, x)

def DerReLU(z):
    return z > 0

def forward_propagation(w1, b1, w2, b2, feature):
    z1 = w1.dot(feature) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)

    return z1, a1, z2, a2

def back_propagation(z1, a1, a2, w2, feature, label):
    m = label.shape[1]

    # Output layer biases and weights
    dz2 = a2 - label
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, 1, keepdims=True)

    # Hidden layer biases and weights
    dz1 = w2.T.dot(dz2) * DerReLU(z1)
    dw1 = 1 / m * dz1.dot(feature.T)
    db1 = 1 / m * np.sum(dz1, 1, keepdims=True)

    return dw1, db1, dw2, db2

def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2

    return w1, b1, w2, b2

def get_mean_absolute_error(a2, labels):
    return np.mean(np.abs(a2 - labels))

def gradient_descent(feature, label, iterations, alpha):
    w1, b1, w2, b2 = init_parameters()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, feature)
        dw1, db1, dw2, db2 = back_propagation(z1, a1, a2, w2, feature, label)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        if i % 10 == 0:
            print("iteration: ", i)
            print("mean average error:", get_mean_absolute_error(a2, label))

    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(feature_train, label_train, iterations=200, alpha=0.1)

match_idx = np.where(info_test[:, 2] == 'Tranmere')[0]
idx = match_idx[len(match_idx) - 1]
# gets info on the latest Tranmere game
print(info_test[idx])

z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, feature_test[:, idx : idx + 1])
print(f"Expected Raw Half Time Result: {info_test[idx, 2]} - {a2[2,0]} - {a2[3, 0]} {info_test[idx, 3]} ")
print(f"Expected Raw Full Time Result: {info_test[idx, 2]} - {a2[0,0]} - {a2[1, 0]} {info_test[idx, 3]} ")
print(f"Expected Half Time Result: {info_test[idx, 2]} - {np.round(a2[2,0])} - {np.round(a2[3, 0])} {info_test[idx, 3]} ")
print(f"Expected Full Time Result: {info_test[idx, 2]} - {np.round(a2[0,0])} - {np.round(a2[1, 0])} {info_test[idx, 3]} ")