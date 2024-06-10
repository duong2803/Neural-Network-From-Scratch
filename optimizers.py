class Optimizer:
    """Base class for optimizers."""

    def __init__(self):
        pass

    def update(self, parameters, gradients):
        """Updates the parameters based on gradients."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate):
        """
        Initializes the SGD optimizer.

        Params
        -------------
        learning_rate: Learning rate for updates.
        """
        pass

    def update(self, parameters, gradients):
        """Updates parameters using SGD."""
        pass
