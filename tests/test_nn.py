from engine.value import Value
from nn.neuron import Neuron
from nn.layer import Layer
from nn.mlp import MLP

def test_neuron_output_range():
    n = Neuron(3)
    out = n([Value(1.0), Value(2.0), Value(-1.0)])
    assert -1 <= out.data <= 1

def test_layer_output_count():
    l = Layer(3, 5)
    out = l([Value(1.0), Value(2.0), Value(-1.0)])
    assert len(out) == 5

def test_mlp_output_shape():
    m = MLP(3, [4, 4, 1])
    out = m([Value(1.0), Value(2.0), Value(-1.0)])
    assert len(out) == 1

def test_mlp_parameters_nonempty():
    m = MLP(3, [4, 4, 1])
    params = m.parameters()
    # 3->4 layer: 4 neurons * (3 weights + 1 bias) = 16
    # 4->4 layer: 4 neurons * (4 weights + 1 bias) = 20
    # 4->1 layer: 1 neuron * (4 weights + 1 bias) = 5
    assert len(params) == 16 + 20 + 5

def test_training_reduces_loss():
    xs = [[Value(2.0), Value(3.0), Value(-1.0)],
          [Value(3.0), Value(-1.0), Value(0.5)]]
    ys = [Value(1.0), Value(-1.0)]

    m = MLP(3, [4, 4, 1])

    def compute_loss():
        preds = [m(x)[0] for x in xs]
        loss = Value(0.0)
        for pred, target in zip(preds, ys):
            diff = pred - target
            loss = loss + diff * diff
        return loss

    first_loss = compute_loss()
    for p in m.parameters():
        p.grad = 0.0
    first_loss.backward()
    for p in m.parameters():
        p.data += -0.05 * p.grad

    for _ in range(20):
        loss = compute_loss()
        for p in m.parameters():
            p.grad = 0.0
        loss.backward()
        for p in m.parameters():
            p.data += -0.05 * p.grad

    final_loss = compute_loss()
    assert final_loss.data < first_loss.data