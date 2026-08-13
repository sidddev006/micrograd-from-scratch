import math
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

    def tanh(self):
        out = (math.exp(2*self.data) - 1)/(math.exp(2*self.data) + 1)
        out = Value(out, (self, ))
        def _backward():
            self.grad += (1 - out.data **2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        order = []
        visited = set()

        def build_order(node):
            if node not in visited:
                visited.add(node)
                if node.parents:
                    for parent in node.parents:
                        build_order(parent)
                order.append(node)

        build_order(self)
        self.grad = 1.0
        for node in reversed(order):
            if node.parents:
                node._backward()
    #def __radd__(self, other):

''''       
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
print(x.grad, z.grad, y.grad)# prints 10.0 3.0 3.0
w2 = z * x
w2.backward()
print(x.grad, z.grad, y.grad)

print(d)
print(d.parents)
print(type(d.parents[0]))
print(a+b)
print(type(a+b))
print(v.data)
print(a.data) #print always returns a str
''''
t = Value(0)
out = t.tanh()
out.backward()
print(out.data, t.grad)

eps = 1e-4
x1 = Value(0.8 + eps)
x2 = Value(0.8 - eps)
numerical_grad = (x1.tanh().data - x2.tanh().data)/(2*eps)

x = Value(0.8)
out = x.tanh()
out.backward()

print("analytical:", x.grad)
print("numerical:", numerical_grad)