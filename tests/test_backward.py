from engine.value import Value

def test_add_backward():
    a = Value(3)
    b = Value(4)
    c = a + b
    c.backward()
    assert a.grad == 1
    assert b.grad == 1

def test_mul_backward():
    a = Value(3)
    b = Value(4)
    c = a*b
    c.backward()
    assert a.grad == 4
    assert b.grad == 3

def test_chain_rule():
    x = Value(3)
    y = Value(4)
    z = x + y
    w = z * x
    w.backward()
    assert x.grad == 10
    assert z.grad == 3
    assert y.grad == 3

def test_gradient_accumulation():
    x = Value(3)
    y = x*x
    y.backward()
    assert y.data == 9
    assert x.grad == 6