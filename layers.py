class Layer:
    """Base class for layers in the network."""
    def __init__(self):
        pass
    
    def forward(self, inputs):
        """Computes the forward pass."""
        pass
    
    def backward(self, gradient):
        """Computes the backward pass."""
        pass

class Dense(Layer):
    """Fully connected layer (dense layer)."""
    def __init__(self, input_size, output_size):
        """
        Initializes the layer with weights and biases.

        Params
        --------------------------------
        input_size: Number of input neurons.
        output_size: Number of output neurons.
        """
        pass
    
    def forward(self, inputs):
        """Computes the forward pass."""
        pass
    
    def backward(self, gradient):
        """Computes the backward pass and updates gradients."""
        pass

Dense()