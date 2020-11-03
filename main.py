################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

from utils import *
from train import *

#Stuff that I imported
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

    # Load the data
    x_train, y_train, x_test, y_test = load_data()

    #reduce size of x_train, y_train:
    x_train, y_train = x_train, y_train

    # Create validation set out of training data.
    split_ratio = round(len(x_train)*0.2)
    x_val, y_val = x_train[:split_ratio], y_train[:split_ratio]
    x_train, y_train = x_train[split_ratio:], y_train[split_ratio:]

    # Any pre-processing on the datasets goes here.
    train_mean=np.mean(x_train, axis=0)
    train_std=np.std(x_train, axis=0)

    def zScore(dataSet):
        dataSet = (dataSet-train_mean)/train_std
        return dataSet

    #z-score the validation and test sets
    x_train=zScore(x_train)
    x_val=zScore(x_val)
    x_test=zScore(x_test)

    #flatten the images (32x32) into single arrays (1024x1)
    x_train = np.array([np.ravel(x) for x in x_train])
    x_val = np.array([np.ravel(x) for x in x_val])
    x_test = np.array([np.ravel(x) for x in x_test])

    #one hot encoded labels
    y_train = one_hot_encoding(y_train, num_classes=10)
    y_val = one_hot_encoding(y_val, num_classes=10)
    y_test = one_hot_encoding(y_test, num_classes=10)
    

    # backprop_check(x_train[:1], y_train[:1], config, 0)

    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results.pkl', data)