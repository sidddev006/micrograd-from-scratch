class Value:
    def __init__(self, data, parents = None):
        self.data = data
        self.parents = parents
        self.grad = 0.0

    def __repr__(self):
        return str(self.data)

    def __add__(self, other):
        out = self.data + other.data
        out = Value(out, (self, other))
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = self.data * other.data
        out = Value(out, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    #def __radd__(self, other):

        
v = Value(5)
a = Value(5)
b = Value(10)
c = Value(15, (a,b))
#print(c.parents)
d = a*b
d.grad = 1.0
d._backward()
print(a.grad, b.grad)#prints 10.0 5.0
e = a+b
e.grad = 1.0
e._backward()
print(a.grad, b.grad) #prints 11.0 6.0
x = Value(3)
y = Value(4)
z = x+y
w = z*x
w.grad = 1.0
w._backward()
z._backward()
print(x.grad, z.grad, y.grad)
'''
print(d)
print(d.parents)
print(type(d.parents[0]))
print(a+b)
print(type(a+b))
print(v.data)
print(a.data) #print always returns a str
'''