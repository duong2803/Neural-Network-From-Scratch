class LossFunction:
    """Base class for loss functions."""
    def __init__(self):
        pass
    
    def forward(self, y_true, y_pred):
        """Computes the forward pass (loss)."""
        pass
    
    def backward(self, y_true, y_pred):
        """Computes the backward pass (gradient of loss)."""
        pass

class MeanSquaredError(LossFunction):
    """Mean Squared Error loss function."""
    def forward(self, y_true, y_pred):
        """Computes the MSE loss."""
        pass
    
    def backward(self, y_true, y_pred):
        """Computes the gradient of the MSE loss."""
        pass