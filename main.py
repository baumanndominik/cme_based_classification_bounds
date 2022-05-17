'''
MNIST classification bounds from the paper "Frequentist bounds for multi-class classification,"
submitted to Advances in Neural Information Processing Systems, 2022.
'''

from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as rbf
import math
import matplotlib.pyplot as plt

def power_func(meas, lambd, n, k, K, gamma):
    arg = np.exp(2.6)**2*rbf(meas, meas, gamma=gamma) - k@np.linalg.solve(K+n*lambd*np.eye(K.shape[0]), k.T)
    if arg < 0:
        print(arg)
        arg = 0.
    return np.sqrt(arg)

def context_conf_int(meas_train, meas_test, cont_train, K, lambd=1e-4, delta=0.05, rkhs_bound=1, gamma=1/((np.exp(7.5))**2)):
    k = np.exp(2.6)**2*rbf(meas_test, meas_train, gamma=gamma)
    prob = cont_train@np.linalg.solve(K + len(meas_train)*lambd*np.eye(len(meas_train)), k.T)
    val = np.max(prob)
    prob_cont = np.where(prob == np.amax(prob))[0][0]
    cg = 1/4*np.sqrt(np.log(np.linalg.det(K + np.max([1, len(meas_train)*lambd])*np.eye(len(meas_train)))) - 2*np.log(delta))
    epsilon = power_func(meas_test, lambd, len(meas_train), k, K, gamma=gamma)*(np.sqrt(rkhs_bound) + cg/np.sqrt(len(meas_train)*lambd))
    ret_unc = 0 
    if val < 0:
        ret_unc = val
    else:
        ret_unc = np.max([0, (val - epsilon)[0, 0]])
    return val, epsilon

train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

num_train = 10000

train_set_array = train_set.data.numpy()
x_train = train_set_array.reshape(*train_set_array.shape[:-2], -1)
x_train = x_train/(0.5*255) - 1
x_train = x_train[0:num_train, :]
test_set_array = test_set.data.numpy()
x_test = test_set_array.reshape(*test_set_array.shape[:-2], -1)
x_test = x_test/(0.5*255) - 1
train_set_array_targets = train_set.targets.numpy()
y_train = train_set_array_targets.reshape(*train_set_array_targets.shape[:-2], -1)
y_train = y_train[0:num_train]
test_set_array_targets = test_set.targets.numpy()
y_test = test_set_array_targets.reshape(*test_set_array_targets.shape[:-2], -1)

y = np.zeros((10, len(y_train)))
for i in range(len(y_train)):
    y[y_train[i], i] = 1

gamma = 1/((np.exp(7.5))**2)
K = np.exp(2.6)**2*rbf(x_train, x_train, gamma=gamma)

val_st = np.zeros(10)
epsilon_st = np.zeros(10)
tested = []
idx = 0
idx_2 = 0
while len(tested) < 10:
    if y_test[idx] not in tested:
        tested.append(y_test[idx])
        test = x_test[idx, :].reshape(1, -1)
        val, epsilon = context_conf_int(x_train, test, y, K, gamma=gamma)
        val_st[idx_2] = val 
        epsilon_st[idx_2] = epsilon
        idx_2 += 1
    idx += 1

def plot_conf_int(ax, x, y, conf):
    ax.set_ylim([-0.5, 1.1])
    for loc, val, bound in zip(x, y, conf):
        ax.plot([loc, loc], [val - bound, val + bound], color='gray')
        ax.plot([loc - 0.25, loc + 0.25], [bound + val, bound + val], color='gray')
        ax.plot([loc - 0.25, loc + 0.25], [val - bound, val - bound], color='gray')
        ax.plot(loc, val, 'o', color='blue')

t = np.arange(len(val_st))

ax = plt.gca()
plot_conf_int(ax, t, val_st, epsilon_st)
plt.show()