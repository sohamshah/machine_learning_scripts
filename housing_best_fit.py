import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

data = sio.loadmat('housing_dataset/housing_data.mat')
x_train = data['Xtrain']
x_validate = data['Xvalidate']
y_train = data['Ytrain']
y_validate = data['Yvalidate']

xmean = x_train.mean(axis=1,keepdims=True)
x_train = x_train - xmean

lambdas = [10** exp for exp in range (-20, 20)]
p = np.random.permutation(x_train.shape[0])
x_train = x_train[p]
y_train = y_train[p]

chunk_size = x_train.shape[0]/10

def find_w(l, x, y):
    identity = np.identity(x.shape[1])
    identity[x.shape[1]-1][x.shape[1]-1] = 0
    return  np.linalg.inv(x.T.dot(x) + l*identity).dot(x.T.dot(y))

best_avg_loss = 10**50
bestlambda = 0
for lmbda in lambdas:
    test_start_ind = 0
    test_end_ind = chunk_size
    total_loss = 0
    for i in range(10):
        xtest_i = x_train[test_start_ind:test_end_ind]
        ytest_i = y_train[test_start_ind:test_end_ind]
        xtrain_i = np.concatenate([x_train[:test_start_ind], x_train[test_end_ind:]])
        ytrain_i = np.concatenate([y_train[:test_start_ind], y_train[test_end_ind:]])
        test_end_ind+= chunk_size
        test_start_ind+= chunk_size

        w = find_w(lmbda, xtrain_i, ytrain_i)
        a = np.mean(ytrain_i)#alpha = mean of y
        am = np.dot(a,np.ones(1944)).reshape(1944,1)
        first = np.dot((xtest_i.dot(w) + am - ytest_i).T,(xtest_i.dot(w) + am - ytest_i))
        second = (lmbda*w).T.dot(w)
        loss = first + second
        loss = loss[0][0]

        total_loss += loss
    if total_loss/10 < best_avg_loss:
        best_avg_loss = total_loss/10
        bestlambda = lmbda
print bestlambda
print best_avg_loss

w = find_w(bestlambda, x_train, y_train)
a = np.mean(y_validate)#alpha = mean of y
am = np.dot(a,np.ones(y_validate.shape[0])).reshape(y_validate.shape[0],1)
rss = np.dot((x_validate.dot(w) + am - y_validate).T,(x_validate.dot(w) + am - y_validate))
print "RSS is:" + str(rss[0][0])
print w

plt.axis([0,9,-60000,50000])
plt.plot(range(1,9),w.T.tolist()[0][:8], 'ro')
plt.show()
