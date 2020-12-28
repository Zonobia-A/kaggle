import csv
from types import new_class
import numpy as np

def loadTrainData():
    l = []
    with open('train.csv') as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)

    l.remove(l[0])
    l = np.array(l)
    label = l[:,0]
    data = l[:,1:]
    return normalizing(toInt(data)), toInt(label)

def toInt(array):
    array = np.mat(array)
    m, n = np.shape(array)
    newArray = np.zeros((m, n))
    for i in range(m): 
        for j in range(n):
            newArray[i, j] = int(array[i, j])
    return newArray

def normalizing(array):
    m, n = np.shape(array)
    for i in range(m):
        for j in range(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array

def loadTestData():
    l = []
    with open('test.csv') as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = np.array(l)
    return normalizing(toInt(data))

def classify():
    pass
