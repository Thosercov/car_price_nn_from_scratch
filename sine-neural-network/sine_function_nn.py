import matplotlib.pyplot as plt
import numpy as np
import constants as c
from layer import Layer
from activation_relu import Activation_ReLU
from activation_linear import Activation_linear
from loss_mean_squared_error import Loss_mean_squared_error
from optimizer_adam import Optimizer_Adam

if __name__ == '__main__':

    np.random.seed(0)

    x_samples = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (c.N_SAMPLES, 1))
    y_samples = np.sin(x_samples) + np.random.normal(loc = 0.0, scale = 0.3, size = (c.N_SAMPLES,1)) #produce some noise on data

    layer1 = Layer(c.N_INPUT_FEATURES, c.N_LAYER_64_NEURONS)
    activation1 = Activation_ReLU()

    layer2 = Layer(c.N_LAYER_64_NEURONS, c.N_LAYER_64_NEURONS)
    activation2 = Activation_ReLU()

    layer3 = Layer(c.N_LAYER_64_NEURONS, c.N_LAYER_OUTPUT)
    activation3 = Activation_linear()




    loss_function = Loss_mean_squared_error()

    optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)

    accuracy_precision = np.std(y_samples) / 250

    for epoch in range(10001):
    
        layer1.forward(x_samples)
        activation1.forward(layer1.output) 

        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        layer3.forward(activation2.output)
        activation3.forward(layer3.output)
    
        data_loss = loss_function.calculate(activation3.output, y_samples)

        regularization_loss = loss_function.regularization_loss(layer1) + loss_function.regularization_loss(layer2) + loss_function.regularization_loss(layer3)

        loss = data_loss + regularization_loss

        predictions = activation3.output
        
        accuracy = np.mean(np.absolute(predictions - y_samples) <
        accuracy_precision)
        
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            'loss: {loss:.3f} (' +
            f'data_loss: {data_loss:.3f}, ' +
            f'reg_loss: {regularization_loss:.3f}), ' +
            f'lr: {optimizer.current_learning_rate}')
        
        # Backward pass
        loss_function.backward(activation3.output, y_samples)
        activation3.backward(loss_function.dinputs)
        layer3.backward(activation3.dinputs)
        activation2.backward(layer3.dinputs)
        layer2.backward(activation2.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)
        
        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.update_params(layer3)
        optimizer.post_update_params()
    
    np.random.seed(1)
    x_test = np.random.uniform(low = 0.0, high = 2 * np.pi, size = (c.N_SAMPLES, 1))

    layer1.forward(x_test)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    plt.scatter(x_samples, y_samples)
    plt.scatter(x_test, activation3.output)
    plt.show()


