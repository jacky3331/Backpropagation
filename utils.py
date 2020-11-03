################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import pickle
import numpy as np
import yaml


def write_to_file(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data():
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    return X_train, y_train, X_test, y_test


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    return np.array([np.insert(np.zeros(num_classes - 1), label, 1) for label in labels])


def find_accuracy(predicted, target):
    """
    Calculates how well the model predicted the images compared with the actual target.
    """
    count=0
    for index in range(len(target)):
        if np.argmax(predicted[index], axis = 0) == np.argmax(target[index], axis = 0):
            count+=1
    return count/len(target)*100

def shuffle_data(dataSetX, dataSetY): 
    """
    Shuffles the data set with the images matching the targets.
    """
    data_x = np.copy(dataSetX)
    data_y = np.copy(dataSetY)

    shuffleIndex=np.random.permutation(len(dataSetX))

    x_train = data_x[shuffleIndex]
    y_train = data_y[shuffleIndex]

    return x_train, y_train

def mini_batches(dataSetX, dataSetY, batch_size):
    """
    Creates mini batches of the data set, and returns them in the format:
    [[batch1], [batch2], [batch3], ..., [batchZ]]
    """
    n_batches = len(dataSetX) // batch_size
    x_train, y_train = shuffle_data(dataSetX, dataSetY)

    dataSetX = []
    dataSetY = []

    for i in range(n_batches):
        mini_x = x_train[i * batch_size: (i + 1)*batch_size]
        mini_y = y_train[i * batch_size: (i + 1)*batch_size]

        dataSetX.append(mini_x)
        dataSetY.append(mini_y)
    
    return dataSetX, dataSetY








