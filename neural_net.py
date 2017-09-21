import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
import sklearn.preprocessing
from scipy import stats
from scipy.special import expit

def inverse_tanh(z):
    return 1 - (np.power(tanh(z),2))

def sigmoid(z):
    z = np.clip(z,-600,600)
    return expit(z)

def sigmoid_prime(z):
    z = np.clip(z,-600,600)
    return sigmoid(z) * (1-sigmoid(z))

def add_bias(X):
    Xlist = X.tolist()
    for row in Xlist:
        row.append(1)
    return np.asarray(Xlist)

class Neural_Network(object):
    def __init__(self):
        #define hyperparams
        self.n_in = 784
        self.n_out = 10
        self.n_hid = 200
        self.V = np.random.normal(loc=0, scale=0.005, size=(self.n_hid, self.n_in + 1))
        #add bias column to V
        self.W = np.random.normal(loc=0, scale=0.005, size=(self.n_out, self.n_hid + 1))
        #add bias column to W
        self.layers = [(1,self.V), (2,self.W)]

    def predict(self, X, Y):
        prediction = self.forward(X)
        self.predicted = []
        for row in prediction[2]:
            self.predicted.append(row.argmax())
        self.true = []
        for row in Y:
            self.true.append(row.argmax())
        print self.predicted
        print 'accuracy is: ' + str(accuracy_score(self.true, self.predicted))


    def forward(self, X):
        outputs = [X] #start with input
        for i, w_i in self.layers:
            last_output = outputs[-1] #initially x = O_i
            if i == 1: #hidden layer, with tanh activation, weight vector V
                #                 1 by 785 x 785 by 200
                inside = np.dot(last_output, w_i.T) # inside = x * V.T
                output = np.tanh(inside) #O_j, 1 by 200
                outputs.append(output)
            else: #output layer with sigmoid activation, weight vector W
                inside = np.dot(add_bias(last_output), w_i.T) # 1 by 201 * 201 by 10
                output = sigmoid(inside) #1 by 10 = O_k
                outputs.append(output)
        return outputs
        #outputs is [X, tanh(x * V.T), S((h + bias) * W.T)]

    def backprop(self, outputs, target_k, loss_func='mse'):
        z_minus_y = outputs[2] - target_k #1 by 10
        delta_k = np.multiply(outputs[2],(1-outputs[2]))
        delta_k = np.multiply(delta_k, z_minus_y)
        dJdW = np.dot(delta_k.T, add_bias(outputs[1])) #10 by 1 * 1 by 201 ==> 10 by 201

        summation = np.dot(delta_k, self.W[:,:-1]) #1 by 200
        delta_j = np.multiply((1 - np.power(outputs[1], 2)), summation)
        dJdV = np.dot(delta_j.T, outputs[0])
        return dJdW, dJdV

    def train(self, X, Y, LR, minibatch_size=1):
        i = 0
        # batches = [bat]
        for sample in X:
            print i
            # sample = X[i]
            sample = sample.reshape(1,785)
            sample_label = Y[i]
            sample_label = sample_label.reshape(1,10)
            dJdW, dJdV = self.backprop(output, sample_label, 'mse')
            self.V = self.V - (LR * dJdV)
            self.W = self.W - (LR * dJdW)
            i += 1
        print 'done training one epoch'
        #update W[index] =  ((W[index]) - learningrate * dJdW)


if __name__ == '__main__':
    data = sio.loadmat('dataset/train.mat')
    train_data = np.swapaxes(data['train_images'],0,2)
    train_data = np.swapaxes(train_data,1,2)
    train_data = train_data.reshape(len(train_data), -1) #shape is 60000, 784
    labels = data['train_labels']
    p = np.random.permutation(train_data.shape[0])
    labels = labels[p]
    train_data = train_data[p]

    new_labels = []
    for row in labels:
        temp = np.zeros(10)
        temp.fill(0.15)
        temp[row[0]] = 0.85
        new_labels.append(temp)
    labels = np.asarray(new_labels)

    #normalize train_data
    train_data_normalized = sklearn.preprocessing.scale(train_data).astype(float)
    norm_list = train_data_normalized.tolist()
    #add bias column to training data
    for row in norm_list:
        row.append(1)
    train_data_normalized = np.asarray(norm_list)

    validation_data = train_data_normalized[50000:]
    train_data = train_data_normalized[:40000]
    validation_labels = labels[50000:]
    train_labels = labels[:40000]


    print 'DONE PREPROCESSING'

    net = Neural_Network()

    learningrate = 0.005
    net.train(train_data, train_labels, learningrate, 250)
    net.predict(validation_data, validation_labels)
