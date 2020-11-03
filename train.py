################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from neuralnet import *
from utils import *
from matplotlib import pyplot as plt
from copy import *


def updated_delta(gamma, previous, current):
    delta = gamma * previous + (1 - gamma) * current
    return delta

def train(x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    return five things -
        training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
        best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []
    best_model = None

    model = NeuralNetwork(config=config)
    learning_rate = config["learning_rate"]
    num_epochs = config["epochs"]
    batch_size = config["batch_size"]
    momentum = config["momentum"]
    gamma = config['momentum_gamma']
    L2_penalty = config["L2_penalty"]
    increase_increment = 0
    epoch_iterated = 0

    for iteration in range(num_epochs):
        x_temp, y_temp = mini_batches(x_train, y_train, config["batch_size"])
        all_minibatches = len(x_temp)
        early_stop = config["early_stop"]
        early_stop_epoch = config["early_stop_epoch"]
        patience = 5  # number of times we can increase loss before we early stop
        epoch_iterated += 1

        previous_w = [0, 0]
        previous_b = [0, 0]
        
        """
        Loops through all the mini batches and update the weights after passing a
        individual mini batch into the backpropagation function.
        """
        for i in range(all_minibatches):
            data_x = x_temp[i]
            data_y = y_temp[i]
            
            predictions, loss = model.forward(data_x, data_y)
            model.backward()
            
            counter = 0
            
            for layer in model.layers:
                if isinstance(layer, Layer):
                    if momentum == True:
                        if counter == 0:
                            layer.d_w = updated_delta(gamma, previous_w[0], layer.d_w)
                            layer.d_b = updated_delta(gamma, previous_b[0], layer.d_b)
                            previous_w[0] = layer.d_w
                            previous_b[0] = layer.d_b
                        
                        else:
                            layer.d_w = updated_delta(gamma, previous_w[1], layer.d_w)
                            layer.d_b = updated_delta(gamma, previous_b[1], layer.d_b)
                            previous_w[1] = layer.d_w
                            previous_b[1] = layer.d_b
                        
                        layer.w = layer.w + learning_rate * layer.d_w / batch_size
                        layer.b = layer.b + learning_rate * layer.d_b / batch_size

                        counter += 1
                    else:
                        layer.w = layer.w + learning_rate * layer.d_w / batch_size
                        layer.b = layer.b + learning_rate * layer.d_b / batch_size
            
            counter = 0

        output_loss, accuracy = test(model, x_valid, y_valid)
        valid_loss.append(output_loss)
        valid_acc.append(accuracy)

        output_loss, accuracy = test(model, x_train, y_train)
        train_loss.append(output_loss)
        train_acc.append(accuracy)

        """
        Early stopping stops iterating through the number of epochs, and breaks out of this
        loop when the loss starts increasing a certain amount of time.
        """
        if early_stop == True and epoch_iterated >= early_stop_epoch:

            if len(valid_loss) >= 2 and abs(valid_loss[-1] - valid_loss[-2]) < 0.001:
                increase_increment += 1
            else:
                increase_increment = 0

            if increase_increment == patience or epoch_iterated == num_epochs:
                print("Stopped at epoch: " + str(epoch_iterated))
                break
    best_model = deepcopy(model)

    """
    The plots below displays what is shown in the title.
    """

    plt.errorbar(range(epoch_iterated), train_loss, label='Train line')
    plt.errorbar(range(epoch_iterated), valid_loss, label='Val line')
    plt.title('Epoch vs Cross Entropy Loss')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.show()

    plt.errorbar(range(epoch_iterated), train_acc, label='Train line')
    plt.errorbar(range(epoch_iterated), valid_acc, label='Val line')
    plt.title('Epoch vs Accuracy')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    return train_acc, valid_acc, train_loss, valid_loss, best_model


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and return loss and accuracy on the test set.
    """
    predictions, loss = model.forward(x_test, y_test)
    accuracy = find_accuracy(predictions, y_test)
    return loss, accuracy

def backprop_check(x_train, y_train, config, layer_index):
    """
    Calculates and prints the difference between the approximation and the gradient after backpropagation.
    """
    model = NeuralNetwork(config=config)
    espilon = 10e-2
    l = model.layers[layer_index]
    weights = l.w[0, 0]
    l.b[0, 0] = weights + espilon
    predictions, loss = model(x_train, y_train)
    firstLoss = loss

    l.b[0, 0] = weights - espilon
    predictions, loss = model(x_train, y_train)
    secondLoss = loss
    
    approximation = np.abs((firstLoss - secondLoss) / (2 * espilon))

    model.backward()
    gradient_calculation = np.abs(l.d_[0, 0])
    
    print("Approximation: " + str(approximation))
    print("Actual: " + str(gradient_calculation))
    print("Difference: " + str(np.abs(gradient_calculation - approximation)))
