class Activation_linear:

        def forward(self, inputs):
            self.inputs = inputs
            self.output = inputs
        
        def backward(self, dvalues):
            self.dinputs = dvalues.copy()