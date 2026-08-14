from engine.value import Value
from nn.mlp import MLP

xs = [[Value(2.0), Value(3.0), Value(-1.0)],
      [Value(3.0), Value(-1.0), Value(0.5)]]

ys = [Value(1.0), Value(-1.0)]

m = MLP(3, [4, 4, 1])

for step in  range(50):
    preds = [m(x)[0] for x in xs]

    loss = Value(0.0)
    for pred, target in zip(preds, ys):
        diff = pred - target
        loss = loss + diff**2

    for p in m.parameters():
        p.grad = 0.0

    loss.backward()

    for p in m.parameters():
        p.data += -0.05 * p.grad
    
    print(step+1, loss.data)