import numpy as np
import scipy.io as sio
import csv
from math import sqrt
from sklearn.feature_extraction import DictVectorizer
from Node import Node
from sklearn.metrics import accuracy_score
import pandas as pd
from SpamDT import DecisionTree, preprocess_census
from random import randint
from scipy import stats
import time, datetime

def bagged(data):
    bagged = []
    for _ in range(data.shape[0]):
        bagged.append(data[randint(0, data.shape[0]-1)])
    return np.array(bagged)


class RandomForest:
    def __init__(self, n_trees):
        self.forest = []
        self.n_trees = n_trees

    def train(self, data, labels):
        depthlimit = 15
        features_per_split = int(sqrt(data.shape[1]))
        for i in range(self.n_trees):
            clf = DecisionTree(data, labels, depthlimit, features_per_split)
            print 'tree trained: ' + str(i)
            self.forest.append(clf)

    def predict(self, data):
        num_rows = data.shape[0]
        i = 1
        predictions = []
        for clf in self.forest:
            predictions.append(clf.predict(data))
            print 'predicted: '+str(i)
            i+=1
        return stats.mode(np.asarray(predictions))[0][0]

if __name__ == '__main__':
    # data = sio.loadmat('spam-dataset/spam_data.mat')
    # spam_train = data['training_data']
    # spam_train_labels = data['training_labels'][0]
    # spam_test = data['test_data']

    #CROSSVALIDATION FOR SPAM
    # train_size = spam_train.shape[0]
    # p = np.random.permutation(train_size)
    # k = 3
    # chunk_size = train_size/3
    # test_start_ind = 0
    # test_end_ind = chunk_size
    # accuracy_total = 0
    # for i in range(3):
    #     xtest_i = spam_train[test_start_ind:test_end_ind]
    #     ytest_i = spam_train_labels[test_start_ind:test_end_ind]
    #     xtrain_i = np.concatenate([spam_train[:test_start_ind], spam_train[test_end_ind:]])
    #     ytrain_i = np.concatenate([spam_train_labels[:test_start_ind], spam_train_labels[test_end_ind:]])
    #
    #     test_end_ind+= chunk_size
    #     test_start_ind+= chunk_size
    #     rf = RandomForest(10)
    #     rf.train(xtrain_i, ytrain_i)
    #     predicted = rf.predict(xtest_i)
    #     acs = accuracy_score(ytest_i, predicted)
    #     print acs
    #     accuracy_total += acs
    # accuracy_total = accuracy_total/k

    # SPAM RANDOMFOREST CSV PREDICTION
    # rf = RandomForest(10)
    # rf.train(spam_train, spam_train_labels)
    # predicted = rf.predict(spam_test)
    # p = [i for i in range(1,len(predicted)+1)]
    # kagg = np.column_stack((p,predicted))
    # np.savetxt("SpamRFtreesBagged.csv", kagg, fmt='%i', delimiter=",")

    #CENSUS PREDICTION CSV
    x, labels = preprocess_census('census_data/train_data.csv')
    test = preprocess_census('census_data/test_data.csv')
    print 'done preprocessing'
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print st
    rf = RandomForest(10)
    rf.train(x, labels)

    print 'done training'
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print st
    predicted = rf.predict(test)
    p = [i for i in range(1,len(predicted)+1)]
    kagg = np.column_stack((p,predicted))
    np.savetxt("CensusRF10Trees.csv", kagg, fmt='%i', delimiter=",")
