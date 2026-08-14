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

xs = [[Value(2.0), Value(3.0), Value(-1.0)],
     [Value(3.0), Value(-1.0), Value(0.5)]]
ys = [Value(1.0), Value(-1.0)]

m = MLP(3, [4, 4, 1])

for step in range(50):
    preds = [m(x)[0] for x in xs]
    loss = Value(0.0)
    for pred, target in zip(preds, ys):
        diff = pred + target * Value(-1)
        loss = loss + diff * diff
    
    params = [p for layer in m.layers for n in layer.neurons for p in (n.w + [n.b])]
    for p in params:
        p.grad = 0.0
    loss.backward()

    for p in params:
        p.data += -0.05 * p.grad
    
    print(step+1, loss.data)