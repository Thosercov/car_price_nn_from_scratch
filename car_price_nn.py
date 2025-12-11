import pandas as pd
import numpy as np
import data_prep as dp
import constants as c
from layer import Layer
from activation_relu import Activation_ReLU
from standardization_z_score import Standardization_Z_score

if __name__ == '__main__':

    inputs = dp.get_inputs() #pandas turns it into numpy.ndarray
    inputs = inputs.drop_duplicates()

    np.random.seed(3)

    layer1 = Layer(c.N_INPUTS, c.N_NEURONS_L1)

    activation1 = Activation_ReLU()

    layer1.forward(inputs)

    activation1.forward(layer1.output)

