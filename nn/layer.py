import sys
import os
import random
from engine.value import Value
from nn.neuron import Neuron

class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.neurons =  [Neuron(num_inputs) for _ in range(num_neurons)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs

