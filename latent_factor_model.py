import numpy as np
import scipy.io as sio
import math
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import euclidean

def MSE(UV, R):
    mse = 0
    for i in range(UV.shape[0]):
        for j in range(UV.shape[1]):
            if not math.isnan(R[i][j]):
                l = UV[i][j] - R[i][j]
                mse += l*l
    return mse

def MSE_dwAboutNan(UV, R):
    mse = UV - R
    mse = np.multiply(mse, mse)
    return mse.sum()



def PCA_SVD(R, og_R, validation_data, d):
    #first center R, compute eigenvectors and values of R.t*R,

    mean = R.mean(axis=0)
    R_center = R - mean
    U,D,V = np.linalg.svd(R_center, full_matrices=False)


    #compute mean squared error loss:
    UV = np.dot(U[:,:d], V[:d,:])
    mse = MSE(UV, og_R)
    print 'MSE for d = ' + str(d) + ' is: ' + str(mse)
    #predict for the validation set
    Predicted_ratings = np.dot(U[:, :d], np.dot(np.diag(D)[:d, :d], V.T[:d, :]))
    predicted = []
    for entry in validation_data:
        if Predicted_ratings[entry[0]-1, entry[1]-1] > 0:
            predicted.append(1)
        else:
            predicted.append(0)
    true_values = validation_data[:,2]
    accuracy = accuracy_score(true_values, predicted)
    print 'Accuracy for d = ' + str(d) + ' is: ' + str(accuracy)


#hyperparameter epsilon, lambda
def gradientDescent(R, eps, lmbda, d, validation_data, og_R):
    n,m = R.shape
    U = np.random.normal(0,0.05,(n,d))
    V = np.random.normal(0,0.05,(d,m))
    prevMSE = MSE_dwAboutNan(np.dot(U,V), R)
    change = 100000
    update_U = True
    while change > 2:
        if update_U:
            dLdU = np.dot(np.dot(U,V) - R, V.T) + np.multiply(U, lmbda)
            U = U - np.multiply(dLdU, eps)
            update_U = False
        else:
            dLdV = np.dot(U.T, (np.dot(U,V) - R)) + np.multiply(V, lmbda)
            V = V - np.multiply(dLdV, eps)
            update_U = True
        newMSE = MSE_dwAboutNan(np.dot(U,V), R)
        change = prevMSE - newMSE
        prevMSE = newMSE
        print 'change is: ' + str(change)
    Predicted_ratings = np.dot(U,V)
########   VALIDATION SET TESTING ##################################
    predicted = []
    for entry in validation_data:
        if Predicted_ratings[entry[0]-1, entry[1]-1] > 0:
            predicted.append(1)
        else:
            predicted.append(0)
    true_values = validation_data[:,2]
    accuracy = accuracy_score(true_values, predicted)
    mse = MSE(Predicted_ratings, og_R)
    print 'MSE for d = ' + str(d) + ' is: ' + str(mse)
    print 'Accuracy for d = ' + str(d) + ' is: ' + str(accuracy)
############ TEST SET PREDICTING #################################
    # new_query_data = []
    # with open('data/joke_data/query.txt', 'r') as query_file:
    #     query_data = query_file.readlines()
    # for entry in query_data:
    #     temp = entry.strip('\n').split(',')
    #     new_query_data.append((int(temp[0]), int(temp[1]), int(temp[2])))
    # query_data = np.asarray(new_query_data)
    # predicted = []
    # p = []
    # for entry in query_data:
    #     if Predicted_ratings[entry[1]-1, entry[2]-1] > 0:
    #         predicted.append(1)
    #     else:
    #         predicted.append(0)
    #     p.append(entry[0])
    # kagg = np.column_stack((p,predicted))
    # np.savetxt("kaggle_submission.txt", kagg, fmt='%i', delimiter=",")


if __name__ == '__main__':
    data = sio.loadmat('data/joke_data/joke_train.mat')
    train_data = data['train']
    #change missing values to 0:
    og_train_data = train_data
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

    users = set(); users_in_valid = validation_data[:,0]
    for i in users_in_valid:
        users.add(i)
    mean = train_data.mean(axis=0)
    R_center = train_data - mean
    for d in [2,5,10,20]:
        #without GD
        PCA_SVD(train_data, og_train_data, validation_data, d)
        #with GD
        gradientDescent(R_center,0.0005,0.05,d, validation_data, og_train_data)
    # gradientDescent(R_center,0.0001,0.01,5, validation_data, og_train_data)
#for ep = 0.0001, labmda - 0.01, d = 20: 0.624629375023
#for ep = 0.0005, labmda - 0.01, d = 20: 0.626287262873
#for ep = 0.0001, labmda - 0.05, d = 20: 0.627371273713
#for ep = 0.0005, labmda - 0.05, d = 20: 0.630352303523
#for ep = 0.00035, labmda - 0.035, d = 20: 0.625474254743
# p = [i for i in range(1,len(predicted)+1)]
# kagg = np.column_stack((p,predicted))
# np.savetxt("CensusRF10Trees.csv", kagg, fmt='%i', delimiter=",")
#  echo 'Id,Category' | cat - kaggle_submission.txt > temp && mv temp kaggle_submission.txt
