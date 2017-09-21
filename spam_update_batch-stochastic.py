import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import preprocessing
import scipy.io as sio

def s(x):
    return 1/(1+math.exp(-x))


data = sio.loadmat('spam_dataset/spam_data.mat')
train = data['training_data']
trainlbl = data['training_labels']


def risk(x, y, w):
    y = y[0]
    risk = 0
    i = 0
    for xi in x:
        ex = s(w.T.dot(xi))
        risk += y[i]*math.log(max(ex,1E-10)) + (1-y[i])*math.log(max(1-ex,1E-10))
        i+= 1
    risk *= -1
    return risk

def batch_update(x, y, w, ep):
    grad_R = 0
    y = y[0]
    i = 0
    w = w.reshape(32,1)
    for xi in x:
        xi = xi.reshape(32,1)
        grad_R += (y[i] - s(xi.T.dot(w)))*xi
        i += 1
    w_new = w + ep*grad_R
    return w_new

def logistic_batch(x, y, ep, label):
    w = np.zeros(32)
    risk_arr = []
    for i in range(1000):
        risk_arr.append(risk(x, y, w))
        w = batch_update(x, y, w, ep)
    plt.plot(risk_arr)
    plt.ylabel(label)
    plt.show()

#q3.2
def stochastic_update(x, y, w, ep):
    grad_R = 0
    y = y[0]
    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    x=x[:200]
    y=y[:200]
    i = 0
    w = w.reshape(32,1)
    for xi in x:
        xi = xi.reshape(32,1)
        w += ep * ((y[i] - s(xi.T.dot(w)))*xi)
        i += 1
        if i >= 200:
            return w

def logistic_stochastic(x, y, ep, label):
    w = np.zeros(32)
    risk_arr = []
    for i in range(1000):
        risk_arr.append(risk(x, y, w))
        w = stochastic_update(x, y, w, ep)
    plt.plot(risk_arr)
    plt.ylabel(label)
    plt.show()

#q3.3
def logistic_stochastic_changing_lr(x, y, label):
    w = np.zeros(32)
    risk_arr = []
    for i in range(1,1001):
        risk_arr.append(risk(x, y, w))
        w = stochastic_update(x, y, w, float(1)/i)
        print i
    plt.plot(risk_arr)
    plt.ylabel(label)
    plt.show()


#PREPROCESSING
x1 = preprocessing.scale(train.astype(float))

x2 = train.astype(float)
for i in xrange(train.shape[0]):
    for j in xrange(train.shape[1]):
        x2[i][j] = math.log(x2[i][j] + 0.1)

x3 = train.astype(float)
for i in xrange(train.shape[0]):
    for j in xrange(train.shape[1]):
        if x3[i][j] > 0:
            x3[i][j] = 1
        else:
            x3[i][j] = 0


logistic_batch(x1, trainlbl, 0.0001, "scaled")
# logistic_batch(x2, trainlbl, 0.0001, "log")
# logistic_batch(x3, trainlbl, 0.0001, "indicator")
# logistic_stochastic(x1, trainlbl, 0.0001, "scaled")
# logistic_stochastic(x2, trainlbl, 0.0001, "log")
# logistic_stochastic(x3, trainlbl, 0.0001, "indicator")
# logistic_stochastic_changing_lr(x1, trainlbl, "scaled")
# logistic_stochastic_changing_lr(x2, trainlbl, "log")
# logistic_stochastic_changing_lr(x3, trainlbl, "indicator")
