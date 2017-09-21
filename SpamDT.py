import numpy as np
import scipy.io as sio
import csv
from sklearn.feature_extraction import DictVectorizer
from Node import Node
from sklearn.metrics import accuracy_score
import pandas as pd

import datetime
import time

def info_gain(left_label_hist, right_label_hist, parent_entropy):

    total_l = left_label_hist[0] + left_label_hist[1]
    total_r = right_label_hist[0] + right_label_hist[1]
    p0_left = 0 if total_l == 0 else float(left_label_hist[0])/total_l
    p1_left = 0 if total_l == 0 else float(left_label_hist[1])/total_l
    p0_right = 0 if total_r == 0 else float(right_label_hist[0])/total_r
    p1_right = 0 if total_r == 0 else float(right_label_hist[1])/total_r

    h_l_0 = 0 if p0_left == 0 else -p0_left*np.log2(p0_left)
    h_l_1 = 0 if p1_left == 0 else -p1_left*np.log2(p1_left)
    h_r_0 = 0 if p0_right == 0 else -p0_right*np.log2(p0_right)
    h_r_1 = 0 if p1_right == 0 else -p1_right*np.log2(p1_right)

    h_l = h_l_0 + h_l_1
    h_r = h_r_0 + h_r_1

    h_after = float(total_l * h_l + total_r * h_r) / (total_r+total_l)
    return parent_entropy - h_after

def zero_and_one_counts(labels):
    zeroCount = 0
    oneCount = 0
    for i in labels:
        if i == 0:
            zeroCount+=1
        elif i == 1:
            oneCount+=1
    return zeroCount, oneCount

def left_and_right(data, labels, xi, split):
    left_1 = 0; left_0 = 0; right_1 = 0; right_0 = 0
    sample_i = 0
    for sample in data:
        if sample[xi]<=split:
            if labels[sample_i] == 1:
                left_1 += 1
            else:
                left_0 += 1
        else:
            if labels[sample_i] == 1:
                right_1 += 1
            else:
                right_0 += 1
        sample_i += 1
    return left_0, left_1, right_0, right_1

def segmenter(data, labels, features_per_split):
    zeroCount, oneCount = zero_and_one_counts(labels)
    p0 = float(zeroCount)/len(labels)
    p1 = float(oneCount)/len(labels)
    parent_entropy = -(p0*np.log2(p0) + p1*np.log2(p1))

    best_info_gain = 0
    best_split_rule = (0, 0)
    if features_per_split is None:
        split_features = range(len(data[0]))
    else:
        p = np.random.permutation(len(data[0]))
        split_features = p[:features_per_split]#pick root(d) features to

    for xi in split_features: #for each feature
        # data_sorted_xi = data[data[:,xi].argsort()]
        # count_dict = {'left_1':0, 'left_0':0, 'right_1':oneCount, 'right_0':zeroCount}
        features = data[:,xi]
        features = np.unique(features)
        if len(features) == 2:
            split = float(features[0] + features[1])/2
            #left_0 is number of samples of class 0 on the left side of split. same applies for others
            left_0, left_1, right_0, right_1 = left_and_right(data, labels, xi, split)
            xi_info_gain = info_gain((left_0,left_1), (right_0,right_1), parent_entropy)
            if float(xi_info_gain) > float(best_info_gain):
                best_info_gain = float(xi_info_gain)
                best_split_rule = (xi, split)
        else:
            #rather than all splits, do 20, 40, 60, 80 percentiles
            #OR use the constant updating technicque
            k = 5
            chunk_size = len(features)/k

            for i in range(1,k):
                split = features[i*chunk_size]
                left_0, left_1, right_0, right_1 = left_and_right(data, labels, xi, split)

                xi_info_gain = info_gain((left_0,left_1), (right_0,right_1), parent_entropy)
                if float(xi_info_gain) > float(best_info_gain):
                    best_info_gain = float(xi_info_gain)
                    best_split_rule = (xi, split)
    return best_split_rule

def classify(root, sample):
    if root.isLeaf():
        return root.getLabel()
    else:
        sr = root.splitRule()
        print 'sample length: ' + str(len(sample))
        print 'sr[0]: ' + str(sr[0])
        print 'sr[1]: ' + str(sr[1])
        if sample[sr[0]] <= sr[1]:
            # print 'less than or equal to'
            return classify(root.leftChild(), sample)
        else:
            # print 'greater than'
            return classify(root.rightChild(), sample)

class DecisionTree(object):
    def __init__(self, data, labels, depthlimit, features_per_split=None):
        self.depthlimit = depthlimit
        self._root = self.grow_tree(data, labels, 0, features_per_split)

    def grow_tree(self, data, labels, maxdepth, features_per_split=None):
        zeroCount, oneCount = zero_and_one_counts(labels)

        if maxdepth >= self.depthlimit or len(labels) <= 20:
            if oneCount > zeroCount:
                return Node(label=1)
            else:
                return Node(label=0)

        if float(zeroCount)/len(labels) >= 0.85:
            return Node(label=0)
        elif float(oneCount)/len(labels) >= 0.85:
            return Node(label=1)

        best_sr = segmenter(data, labels, features_per_split)

        # print best_sr
        S_l = []
        labels_l = []
        S_r = []
        labels_r = []
        sample_i = 0
        for sample in data:

            if sample[best_sr[0]] <= best_sr[1]:
                S_l.append(sample)
                labels_l.append(labels[sample_i])
            else:
                S_r.append(sample)
                labels_r.append(labels[sample_i])
            sample_i+=1
        S_l = np.array(S_l)
        S_r = np.array(S_r)

        return Node(left=self.grow_tree(S_l, labels_l, maxdepth + 1, features_per_split), right=self.grow_tree(S_r, labels_r, maxdepth + 1, features_per_split), split_rule=best_sr)


    def predict(self, data):
        predicted = []
        root = self._root
        for sample in data:
            predicted.append(classify(root, sample))
            # return None
        return predicted


def preprocess_census(name):
    D = []
    labels = []
    csvfile = open(name)

    majority_of_each_feature = {}
    df = pd.read_csv(name)
    for col in df.columns:
        feature_col = df[col]
        majority_of_each_feature[col] = feature_col.mode()[0]


    #edit ? entries
    reader = csv.DictReader(csvfile)
    for row in reader:
        D.append(row)
    for entry in D:
        for key in entry:
            if entry[key] == '?':
                entry[key] = majority_of_each_feature[key]


    #binarize categorical features
    for e in D:
        e['age'] = int(e['age'])
        e['fnlwgt'] = int(e['fnlwgt'])
        e['capital-gain'] = int(e['capital-gain'])
        e['capital-loss'] = int(e['capital-loss'])
        e['education-num'] = int(e['education-num'])
        e['hours-per-week'] = int(e['hours-per-week'])
        if (e.has_key('label')):
            lbl = e.pop('label')
            labels.append(int(lbl))

    v = DictVectorizer(sparse=False)
    X = v.fit_transform(D)
    csvfile.close()
    if (len(labels) == 0):
        return X
    return X, labels



if __name__ == '__main__':
    data = sio.loadmat('spam-dataset/spam_data.mat')
    spam_train = data['training_data']
    spam_train_labels = data['training_labels'][0]
    spam_test = data['test_data']
        # USE BELOW CODE FOR 3 FOLD CROSS VALIDATION
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
    #     print xtrain_i.shape
    #     print ytrain_i.shape
    #     print chunk_size
    #     test_end_ind+= chunk_size
    #     test_start_ind+= chunk_size
    #     dt = DecisionTree(data=xtrain_i, labels=ytrain_i, depthlimit=20)
    #     predicted = dt.predict(xtest_i)
    #     acs = accuracy_score(ytest_i, predicted)
    #     print acs
    #     accuracy_total += acs
    # accuracy_total = accuracy_total/k

    # SPAM PREDICTION CSV
    # dt = DecisionTree(data=spam_train, labels=spam_train_labels, depthlimit=20)
    # predicted = dt.predict(spam_test)
    # p = [i for i in range(1,len(predicted)+1)]
    # kagg = np.column_stack((p,predicted))
    # np.savetxt("SpamDT.csv", kagg, fmt='%i', delimiter=",")


    #CENSUS PREDICTION CSV
    x, labels = preprocess_census('census_data/train_data.csv')
    test = preprocess_census('census_data/test_data.csv')
    print 'done preprocessing'
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print st
    dt = DecisionTree(data=x, labels=labels, depthlimit=20)
    print 'done training'
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print st
    predicted = dt.predict(test)
    p = [i for i in range(1,len(predicted)+1)]
    kagg = np.column_stack((p,predicted))
    np.savetxt("CensusDT.csv", kagg, fmt='%i', delimiter=",")

# k=10
#census train start: 2016-04-12 19:20:36
#census train end:   2016-04-12 19:22:24

# k=5
#census train start: 2016-04-12 19:23:32
#census train end:   2016-04-12 19:24:30
