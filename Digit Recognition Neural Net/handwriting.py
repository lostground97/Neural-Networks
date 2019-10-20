#Written by Rishabh Sethi.
#http://neuralnetworksanddeeplearning.com/chap1.html
import numpy as np
import random
import mnist_loader
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

#### Load the MNIST data
def load_data_shared(filename="mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

class Network(object):
    #sizes is an arrray of numbers containing number of neurons at each layer
    #len(sizes) gives number of layers user wants. Suppose sizes=[2,3,4,1] It means that the user wants 4 layers with i/p layers with 2 nodes and output layer with 1 node. Moreover, 2 hidden layers with 3 and 4 nodes respectively.
    #We need biases for each layer. So we traverse the size array and set random bias for that particular layer.
    #Then weights need to be assigned for each path in NUeral Net. So we need a 2-d matrix thus x and y will random weights to all paths in network.
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        #sizes[1:] assumes that the first layer is input layer therefore removes the bias from that layer.
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        #np.random.randn generates Gaussian Distribution with mean 0 and standard deviation 1.
    
    #np.dot is a scalar multiplication of 2 vectors.
    def feedforward(self,a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def update_mini_batch(self, batch, eta):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            del_new_b, del_new_w = self.backprop(x, y)
            new_b = [nb + dnb for nb, dnb in zip(new_b, del_new_b)]
            new_w = [nw + dnw for nw, dnw in zip(new_w, del_new_w)]
            self.biases = [b - (eta/len(batch))*nb for b, nb in zip(new_b, del_new_b)]
            self.weights = [w - (eta/len(batch))*nw for w, nw in zip(new_w, del_new_w)]

    #Training data is collection of pairs (X,Y) such that the X contains the input and Y is the desired output.
    #Since this is training function therefore test_data=None otherwise it would have been test_data.
    #eta is the learning rate of the StochasticGradientDescent Algorithm.
    def StochasticGradientDescent(self, training_data, epochs, batch_size, eta, test_data=None):
        n_test = 0
        if(test_data):
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0,n,batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if(test_data):
                print("Epoch 0/1/2".format(j, evaluate(test_data), n_test))
            else:
                print("Complete Epoch".format(j))



#We use np.exp instead of normal exponent function because whenever x is an vector, numpy automatically applies the sigmoid function elementwise.
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


net = Network([2,3,4])
#print(net.biases)
#print(net.weights)
#net.weights[1] shows the weights of all paths from layer 2 to 3. Moreover net.weights[1][2][3] shows the weight between 2nd neuron of 2nd layer and 3rd neuron of 3rd layer.