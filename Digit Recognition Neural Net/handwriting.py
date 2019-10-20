#Written by Rishabh Sethi.
#http://neuralnetworksanddeeplearning.com/chap1.html
import numpy as np

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

#net = Network([2,3,4])
#print(net.biases)
#print(net.weights)