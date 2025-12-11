import numpy as np

class Standardization_Z_score:

    def __init__(self, inputs):
        self.inputs = inputs
        self.standard_deviation = np.std(inputs, axis=0) #population standard deviation 
        self.mean = np.array(inputs.mean().values)

    def standardize(self):
        for row in self.inputs:
            for column in row:
                pass # here make the standardization code
