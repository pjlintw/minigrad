

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'

x = Value(3.0)
print(x)