import sys
import os
import random
from engine.value import Value

class Neuron:
    def __init__(self, num_inputs):
        #w -> weights
        #b -> bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        total = self.b
        for xi, wi in zip(x, self.w):
            total = total + xi * wi
        out = total.tanh()
        return out

n = Neuron(2)
out = n([Value(1.0), Value(2.0)])
print(out)