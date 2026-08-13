class Value:
    def __init__(self, data, parents = None):
        self.data = data
        self.parents = parents
    def __repr__(self):
        return str(self.data)
    def __add__(self, other):
        out = self.data + other.data
        out = Value(out, (self, other))
        return out
    def __mul__(self, other):
        out = self.data * other.data
        out = Value(out, (self, other))
        return out
    #def __radd__(self, other):

        
v = Value(5)
a = Value(5)
b = Value(10)
c = Value(15, (a,b))
#print(c.parents)
d = a*b
print(d)
print(d.parents)
print(type(d.parents[0]))
'''
print(a+b)
print(type(a+b))
print(v.data)
print(a.data) #print always returns a str
'''