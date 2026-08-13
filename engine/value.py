class Value:
    def __init__(self, data):
        self.data = data
    def __add__(self, other):
        out = self.data + other.data
        return out
        
v = Value(5)
a = Value(5)
b = Value(10)
print(type(a+b))
print(v.data)