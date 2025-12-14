import matplotlib.pyplot as plt
import numpy as np
import constants as c
from layer import Layer
from activation_relu import Activation_ReLU
from activation_linear import Activation_linear

if __name__ == '__main__':

    np.random.seed(0)

    x_samples = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (c.N_SAMPLES, 1))
    y_samples = np.sin(x_samples) + np.random.normal(loc = 0.0, scale = 0.3, size = (c.N_SAMPLES,1)) #produce some noise on data

    # layer 1
    layer1 = Layer(c.N_INPUTS, c.N_LAYER_1_NEURONS)
    layer1.forward(x_samples)

    activation1 = Activation_ReLU()
    activation1.forward(layer1.output)

    #layer 2
    layer2 = Layer(c.N_LAYER_1_NEURONS, c.N_LAYER_2_NEURONS)
    layer2.forward(activation1.output)

    activation2 = Activation_ReLU()
    activation2.forward(layer2.output)

    # layer 3
    layer3 = Layer(c.N_LAYER_2_NEURONS, c.N_LAYER_3_NEURONS)
    layer3.forward(activation2.output)

    activation3 = Activation_ReLU()
    activation3.forward(layer3.output)

    # layer 4
    layer4 = Layer(c.N_LAYER_3_NEURONS, c.N_LAYER_4_NEURONS)
    layer4.forward(activation3.output)

    activation4 = Activation_ReLU()
    activation4.forward(layer4.output)

    print(activation4.output)


