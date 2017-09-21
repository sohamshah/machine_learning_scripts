import numpy as np
import scipy.io as sio
import math
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot  as plt


def get_data(data, means):
    cluster_set = []
    for i in range(len(means)):
        cluster_set.append([])
    for image in data:
        dist = []
        for i in range(len(means)):
            dist.append(np.linalg.norm(image-means[i]))
        min_index = dist.index(min(dist))
        cluster_set[min_index].append(image)
    return cluster_set

def get_means(cluster_set):
    means = []
    for i in range(len(cluster_set)):
        means.append(np.mean(cluster_set[i], axis=0))
    return means

def kmean_cluster(data, k):
    n, m = data.shape
    means = data[np.random.choice(n,k,replace=False),:]
    i = 0
    while i < 100:
        cluster_set = get_data(data, means)
        means = get_means(cluster_set)
        i += 1
        print i
    error = 0
    for j in range(len(means)):
        error += np.sum(np.power(np.array(cluster_set[j])-means[j], 2))
    print "error is " + str(error)
    return means, error

if __name__ == '__main__':
    data = sio.loadmat('data/mnist_data/images.mat')
    images = data['images']
    images = images.astype('float64')
    images = images.reshape(-1, images.shape[-1]).T
    cluster_centers = [5, 10, 20]
    for k in cluster_centers:
        means, err = kmean_cluster(images, k)
        means = np.asarray(means)
        means = means.swapaxes(0,1).reshape((28, 28, k))
        for i in range(k):
            plt.imshow(means[:,:,i])
            tit = "cluster center = "+str(i) +"; k = "+str(k)
            plt.title(tit)
            plt.show()
        for i in range(2): #checking if loss changes for different iterations
            m, err = kmean_cluster(images, k)
            print "k-mean loss for iteration " + str(i) + ", k = " + str(k) + " is: " + str(err)
