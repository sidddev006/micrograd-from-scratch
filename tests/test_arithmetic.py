from engine.value import Value
def test_addition():
    a = Value(2)
    b = Value(5)
    assert (a+b).data == 7

def test_multiplication():
    a = Value(7)
    b = Value(4)
    assert (a*b).data == 28

def test_subtraction():
    a = Value(7)
    b = Value(5)
    assert (a-b).data == 2

def test_division():
    a = Value(4)
    b = Value(2)
    assert (a/b).data == 2

def test_power():
    a = Value(5)
    assert (a ** 2).data == 25

def test_raw_number_operations():
    a = Value(5)
    assert (a+3).data == 8
    assert (3+a).data == 8
    assert (a*2).data == 10
    assert (2*a).data == 10