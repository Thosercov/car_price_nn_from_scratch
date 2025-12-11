import pandas as pd
import numpy as np
import data_prep as dp
import constants as c
from layer import Layer
from activation_relu import Activation_ReLU
from activation_linear import Activation_linear
from standardization_z_score import Standardization_Z_score

if __name__ == '__main__':

    inputs = dp.get_inputs() #pandas turns it into numpy.ndarray
    inputs = inputs.drop_duplicates()
    standardization_Z_score = Standardization_Z_score()
    standardized_inputs = standardization_Z_score.standardize(inputs)

    np.random.seed(0)

    layer1 = Layer(c.N_INPUTS, c.N_NEURONS_L1)
    activation1 = Activation_ReLU()
    layer1.forward(standardized_inputs)
    activation1.forward(layer1.output)

    layer2 = Layer(c.N_NEURONS_L1, c.N_NEURONS_L2)
    activation2 = Activation_linear()
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    print(activation1.output)



    
