import numpy as np

class Standardization_Z_score:     

    def standardize(self, inputs):
        self.standard_deviation = np.array(np.std(inputs, axis=0).values) #population standard deviation 
        self.mean = np.array(inputs.mean(axis=0).values)
        return (inputs - self.mean) / self.standard_deviation

