# micrograd-from-scratch

A scalar-valued automatic differentiation engine and small neural network library, built entirely from first principles in Python — inspired by Andrej Karpathy's *micrograd*, implemented independently after watching the lecture, without referencing the original source code.

It implements reverse-mode autodiff over a dynamically built computational graph, then composes that engine into `Neuron → Layer → MLP` abstractions capable of training a small feedforward network via gradient descent.

## Architecture

```
Value (scalar wrapper)
    ↓
Operations (+, -, *, /, **, tanh — each returns a new Value)
    ↓
Computational Graph (implicit, built from Value references)
    ↓
Local Derivative Storage (per-node, via closures)
    ↓
Topological Ordering
    ↓
Backpropagation Engine (.backward())
    ↓
Neuron (weighted sum of Values + tanh activation)
    ↓
Layer (collection of Neurons)
    ↓
MLP (stack of Layers)
    ↓
Loss Function (Value-producing comparison)
    ↓
Training Loop (forward → backward → update → zero-grad)
```

## Project structure

```
micrograd-from-scratch/
├── engine/
│   └── value.py          # Value class: ops, graph, backward()
├── nn/
│   ├── neuron.py
│   ├── layer.py
│   └── mlp.py
├── tests/
│   ├── test_arithmetic.py
│   ├── test_backward.py
│   ├── test_grad_check.py
│   └── test_nn.py
├── examples/
│   └── train_toy_dataset.py
└── pytest.ini
```

## Usage

```python
from engine.value import Value
from nn.mlp import MLP

# build a 3-input MLP with two hidden layers of 4 neurons and a 1-neuron output
m = MLP(3, [4, 4, 1])

x = [Value(2.0), Value(3.0), Value(-1.0)]
out = m(x)
print(out)  # [Value(...)]

# gradients flow through the whole network automatically
loss = out[0] ** 2
loss.backward()
print(m.layers[0].neurons[0].w[0].grad)
```

See `examples/train_toy_dataset.py` for a full training loop on a toy regression dataset.

## Core engine capabilities

- Operators: `+`, `-`, `*`, `/`, `**` (constant exponent), unary negation
- Activation: `tanh`
- Automatic graph construction — every `Value` tracks the parents it was created from
- Reverse-mode automatic differentiation via a `.backward()` call that:
  - builds a topological ordering of the graph automatically (no manual ordering required)
  - correctly accumulates gradients across branching/reused nodes
- Raw Python numbers are handled transparently on either side of an operator (`a + 3` and `3 + a` both work)

## Verification

Every piece of this engine is checked two independent ways: hand-derived gradients checked against the code, and numerical gradient checking (finite differences) checked against the analytical backward pass — so correctness isn't just assumed, it's tested.

```bash
pip install pytest
pytest tests/ -v
```

18 tests covering forward arithmetic, single- and multi-path backpropagation, gradient accumulation, numerical gradient checks (including a composite branching expression), and full neuron/layer/MLP/training behavior.

## How this was built

This was built as a deliberate learning exercise: after watching Karpathy's micrograd lecture, I rebuilt the system from a milestone-by-milestone specification — scalar representation → forward ops → graph structure → local derivatives → single-path backprop → topological ordering → full backprop with accumulation → neural network abstractions — without looking at the reference implementation. Each milestone was implemented, tested against hand-derived math, and debugged independently before moving to the next.

## License

MIT