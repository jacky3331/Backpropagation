################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import numpy as np
import math


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.
    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError("%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None
        self.temp = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """

        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """

        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """

        self.x = 1 / (1 + np.exp(-x))
        return self.x

    def tanh(self, x):
        """
        Implement tanh here.
        """

        self.x = np.tanh(x)
        return self.x

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        self.x = np.maximum(x, 0)

        return self.x
        
    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """

        return self.sigmoid(self.x)*(1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """

        return (1 - (np.tanh(self.x))**2)

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        grad = np.ones(self.x.shape)

        return grad

class Layer:
    """
    This class implements Fully Connected layers for your neural network.
    Example:
        >>> fully_connected_layer = Layer(1024, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units,
                                                           out_units)  # You can experiment with initialization.
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """

        self.x = x

        a = np.dot(x, self.w) + self.b
        self.a = a

        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        d_x = (np.dot(delta, self.w.T))
        self.d_x = d_x

        d_b = np.sum(delta, axis = 0)
        self.d_b = d_b

        d_w = (np.dot(self.x.T, delta))
        self.d_w = d_w
        
        return self.d_x


class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.
    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        self.learning_rate = config["learning_rate"]

        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        self.targets = targets
        input_var = x

        for i in range(len(self.layers)):
            l = self.layers[i]
            output = l(input_var)
            input_var = output
        
        self.y = output

        if(targets is not None):
            predictions = self.softmax(self.y)
            loss = self.loss(predictions, targets)
            self.y = predictions
            return predictions, loss

        return self.y

    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        error = (self.targets - self.y) 

        for i in range(len(self.layers) - 1, -1, -1): 
            l = self.layers[i]
            dx = l.backward(error)
            error = dx
            
    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """
        x -= np.max(x)
        exp_X = np.exp(x)
        normalization = np.sum(exp_X, axis=1, keepdims=True)

        return exp_X / normalization

    def loss(self, logits, targets):
        """
        compute the categorical cross-entropy loss and return it.
        """
        logits = np.log(logits + 1e-6)
        temp = targets * logits
        sum = -np.sum(temp, axis=1)
        mean = np.mean(sum)

        return mean
