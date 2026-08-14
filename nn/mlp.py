import sys
import os
import random
from engine.value import Value
from nn.layer import Layer
from nn.neuron import Neuron

class MLP:
    def __init__(self, num_inputs, layer_sizes):
        sizes = [num_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(layer_sizes))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for n in layer.neurons for p in (n.w + [n.b])]

