import pandas as pd
import numpy as np
import data_prep as dp
import constants as c
from layer import Layer

if __name__ == '__main__':
    inputs = dp.get_inputs() #pandas turns it into numpy.ndarray
    inputs = inputs.drop_duplicates()

    np.random.seed(0)

    layer1 = Layer(c.N_INPUTS, c.N_NEURONS_L1)

    layer1.forward(inputs)

    print(layer1.output)




