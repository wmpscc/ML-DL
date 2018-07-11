import pandas as pd
import numpy as np
import random

def load_data(path, rate = 0.8):
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

def euclideanDistance(instance1, instance2):
    ''' Calculate two-point Euclidean distance
    Args:
        instance1: Numpy array type training set.
        instance2: The element of test set.
    Returns:
        distance: The distance of instance2 to instance1, the type is list.
    '''
    distance = []
    for instance in instance1:
        temp = np.subtract(instance, instance2)
        temp = np.power(temp, 2)
        temp = temp.sum()
        distance.append(np.sqrt(temp))
    return distance

def getNeighbors(distance, label, k):
    ''' Calculate k-nearest element
    Args:
        distance: Return by euclideanDistance function.
        label: Training label.
        k: k-nearest
    Returns:
        neighbors: The k-nearest elements.
    '''
    df = pd.DataFrame()
    df['distance'] = distance
    df['label'] = label
    neighbors = df.sort_values(by='distance').reset_index()
    return neighbors.loc[:k-1]

def getResult(neighbors):
    neighbors = neighbors['label'].value_counts()
    neighbors = pd.Series(neighbors).sort_values(ascending=False).reset_index()
    return neighbors['index'][0]

def getAccuracy(testData, testLabel, trainSet, trainLabel):
    ''' Calculate accuracy percentage.
    Args:
        testData: Numpy array of test set.
        testLabel: Numpy array of test labels.
        trainSet: Numpy array of train set.
        trainLabel: Numpy array of train labels.
    Returns:
        score/len(trainLabel)
    '''
    score = 0
    i = 0
    for instance in testData:
        distance = euclideanDistance(trainSet, instance)
        neighbors = getNeighbors(distance, trainLabel, 5)
        result = getResult(neighbors)
        if result == testLabel[i]:
            score += 1
        i += 1
    return score/len(trainLabel)

if  __name__ == "__main__":
    trainData, trainLabel, testData, testLabel = load_data("irisdata.txt", 0.67)
    print(getAccuracy(np.array(testData), np.array(testLabel), np.array(trainData), np.array(trainLabel)))
