import numpy as np
import scipy.io as sio
import math
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import euclidean


def averageRating_recommender(train_data, validation_data):
    joke_ratings = []
    for i in range(100):
        joke_ratings.append(np.average(train_data[:,i]))
    predicted = []
    for entry in validation_data:
        if joke_ratings[entry[1]-1] > 0:
            predicted.append(1)
        else:
            predicted.append(0)
    true_values = validation_data[:,2]
    return accuracy_score(true_values, np.asarray(predicted))

def recommender_kNN(k, users, train_data, validation_data):
    index = 1
    train_tuples = []
    for entry in train_data:
        train_tuples.append((index, entry))
        index += 1
    averaged_knn_jokeRatings = []
    for i in users:
        # print 'user number: '+str(i)
        knn = kNN(k, i, train_data, train_tuples) #includes both the user and its k nearest neighbors
        knn = knn - 1 #adjusting indices to access train_data correctly
        knn = [ int(x) for x in knn ]
        knn_vectors = train_data[knn]
        knn_vectors = knn_vectors.sum(axis=0)
        knn_vectors = knn_vectors.reshape(1,100)
        averaged_knn_jokeRatings.append(np.divide(knn_vectors, k+1))
    ratings = np.asarray(averaged_knn_jokeRatings).reshape(100,100)
    predicted = []
    for entry in validation_data:
        if ratings[entry[0]-1, entry[1]-1] > 0:
            predicted.append(1)
        else:
            predicted.append(0)
    true_values = validation_data[:,2]
    return accuracy_score(true_values, np.asarray(predicted))


#returns array containing the k nearest neighbors of user, where user is the index of certain user
def kNN(k, user, train_data, train_tuples):
    distances = []
    for usr_tuple in train_tuples:
        dist = euclidean(train_data[user-1], usr_tuple[1])
        distances.append((usr_tuple[0], dist))
    distances.sort(key = lambda x: x[1])
    k_closest_user_and_distance = distances[:k+1] # to change to only k nearest neighbors set to --> distances[1:k+1]
    k_closest_users = np.asarray(k_closest_user_and_distance)[:,0]
    return k_closest_users


if __name__ == '__main__':
    data = sio.loadmat('data/joke_data/joke_train.mat')
    train_data = data['train']
    #change missing values to 0:
    train_list = train_data.tolist()
    new_train = []
    for user in train_list:
        temp = []
        for joke_rating in user:
            if math.isnan(joke_rating):
                temp.append(float(0))
            else:
                temp.append(joke_rating)
        new_train.append(temp)
    train_data = np.asarray(new_train)

    new_validation_data = []
    with open('data/joke_data/validation.txt', 'r') as validation_file:
        validation_data = validation_file.readlines()
    for entry in validation_data:
        temp = entry.strip('\n').split(',')
        new_validation_data.append((int(temp[0]), int(temp[1]), int(temp[2])))
    validation_data = np.asarray(new_validation_data)
    #question 2.2: average postive rating:
    accuracy_from_average = averageRating_recommender(train_data, validation_data)
    print('accuracy from average ratings per joke: ' + str(accuracy_from_average))
    #accuracy for average rating is 0.62032520325203255

    #question 2.2: KNN
    users = set(); users_in_valid = validation_data[:,0]
    for i in users_in_valid:
        users.add(i)
    k_arr = [10, 100, 1000]
    for k in k_arr:
        accuracy_from_knn = recommender_kNN(k, users, train_data, validation_data)
        print 'kNN accuracy for ' + str(k) + ' is ' + str(accuracy_from_knn)ß
    #accuracy for 10 is 0.649051490515
    #accuracy for 100 is 0.689430894309
    #accuracy for 1000 is 0.694037940379
