import numpy as np


class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + num_hidden + [num_outputs]

        # initiate random weights
        self.weights = []
        for i in range(len(layers) - 1):
            # for the rows the current layer and for the columns the subsequent layer
            w = np.random.rand(layers[i], layers[i + 1])
