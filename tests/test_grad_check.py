from engine.value import Value

def numerical_gradient(f, x_val, eps = 1e-4):
    x1 = Value(x_val + eps)
    x2 = Value(x_val - eps)
    return (f(x1).data - f(x2).data) / (2*eps)

def test_tanh_gradient_check():
    x_val = 0.8
    x = Value(x_val)
    out = x.tanh()
    out.backward()

    numerical = numerical_gradient(lambda v: v.tanh(), x_val)
    assert abs(x.grad - numerical) < 1e-6

def test_mul_gradient_check():
    def f(v):
        y = Value(5)
        return v * y

    x_val = 3.0
    x = Value(x_val)
    out = f(x)
    out.backward()

    numerical = numerical_gradient(f, x_val)
    assert abs(x.grad - numerical) < 1e-6

def test_composite_gradient_check():
    def f(v):
        return v*v +v
    
    x_val = 2.0
    x = Value(x_val)
    out = f(x)
    out.backward()
    numerical = numerical_gradient(f, x_val)
    assert abs(x.grad - numerical) < 1e-6