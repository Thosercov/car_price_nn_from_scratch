import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        #self.weights[0:8, 0:3] = -np.abs(self.weights[0:8, 0:3]) -> set weights of 8 numerical features to all positive/negatve -> for testing purposes

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases