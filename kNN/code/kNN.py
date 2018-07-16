import pandas as pd
import numpy as np
import random
from collections import Counter


def load_data(path, rate=0.8):
    ''' Making dataSet

    Args:
        path: DataSet file path.
        rate: TrainSet percentage.
    Returns:
        trainSet: Training set.
        trainLabel: Training label.
        testSet: Test set.
        testLabel: Test label.
    '''
    data = pd.read_csv(path)
    randIndex = random.sample(range(0, len(data)), len(data))
    trainSet = data.loc[randIndex[:int(len(data) * rate)]]
    testSet = data.loc[randIndex[int(len(data) * rate):]]
    trainLabel = trainSet['name']
    testLabel = testSet['name']
    trainSet.drop('name', axis=1, inplace=True)
    testSet.drop('name', axis=1, inplace=True)
    return np.array(trainSet), np.array(trainLabel), np.array(testSet), np.array(testLabel)


def euclideanDistance_two_loops(train_X, test_X):
    ''' Calculate two-point Euclidean distance

    Args:
        train_X: Numpy array type training set.
        test_X: Numpy array type test set.
    Returns:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training point.
    '''

    num_test = test_X.shape[0]
    num_train = train_X.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            test_line = test_X[i]
            train_line = train_X[j]
            temp = np.subtract(test_line, train_line)
            temp = np.power(temp, 2)
            dists[i][j] = np.sqrt(temp.sum())
    return dists


def euclideanDistance_no_loops(train_X, test_X):
    '''Calculate two-point Euclidean distance without loops

    Input / Output: Same as euclideanDistance_two_loops
    '''

    num_test = test_X.shape[0]
    num_train = train_X.shape[0]

    sum_train = np.power(train_X, 2)
    sum_train = sum_train.sum(axis=1)
    sum_train = sum_train * np.ones((num_test, num_train))

    sum_test = np.power(test_X, 2)
    sum_test = sum_test.sum(axis=1)
    sum_test = sum_test * np.ones((1, sum_train.shape[0]))
    sum_test = sum_test.T

    sum = sum_train + sum_test - 2 * np.dot(test_X, train_X.T)
    dists = np.sqrt(sum)
    return dists


def l1_distance_no_loops(train_X, test_X):
    '''Calculate two-point L1 distance
    Args:
        train_X: Numpy array type training set.
        test_X: Numpy array type test set.
    Returns:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training point.
    '''

    num_test = test_X.shape[0]
    num_train = train_X.shape[0]

    test = np.tile(test_X, (num_train, 1, 1))
    train = np.tile(train_X, (num_test, 1, 1))
    train = np.transpose(train, axes=(1, 0, 2))

    sum = np.subtract(test, train)
    sum = np.abs(sum)
    sum = np.sum(sum, axis=2)
    dists = sum.T
    return dists


def predict_labels(dists, labels, k=1):
    ''' To predict a label for each test point.
    Args:
        dists: a numpy array of shape (num_test, num_train) where dists[i, j].
        labels: each test point real label.

    Returns:
        y_pred: knn predict target label.
    '''

    num_test = dists.shape[0]
    y_pred = []
    for i in range(num_test):
        index = np.argsort(dists[i])
        index = index[:k]
        closest_y = labels[index]
        name, _ = Counter(closest_y).most_common(1)[0]
        y_pred.append(name)
    return y_pred


def getAccuracy(y_pred, y):
    ''' Calculate the predict accuracy.
    Args:
        y_pred: knn predict target label.
        y: each test point real label.

    Returns:
        accuracy: average accuracy in test set.
    '''
    num_correct = np.sum(y_pred == y)
    accuracy = float(num_correct) / len(y)
    return accuracy


if __name__ == "__main__":
    trainData, trainLabel, testData, testLabel = load_data("irisdata.txt", 0.67)
    dists = l1_distance_no_loops(trainData, testData)
    y_pred = predict_labels(dists, trainLabel, k=3)
    accuracy = getAccuracy(y_pred, testLabel)
    print(accuracy)
