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

l = Layer(2,3)
out = l([Value(1.0), Value(2.0)])
print(out)