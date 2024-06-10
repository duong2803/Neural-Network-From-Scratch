class ActivationFunction:
    """
        Base class for activation functions.
    """
    def __init__(self):
        pass

    def forward(self):
        """
            Computes the forward pass.
        """
        pass

    def backward(self):
        """
            Computes the backward pass (gradient).
        """
        pass

class Sigmoid(ActivationFunction):
    """
        Sigmoid activation function.
        """
    def forward(self, x):
        """
            Computes the sigmoid activation.
        """
        pass
    
    def backward(self, x):
        """
            Computes the gradient of the sigmoid activation.
        """
        pass

class ReLU(ActivationFunction):
    """
        ReLU activation function.
    """
    def forward(self, x):
        """
            Computes the ReLU activation.
        """
        pass
    
    def backward(self, x):
        """
            Computes the gradient of the ReLU activation.
        """
        pass