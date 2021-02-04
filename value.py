
"""Construct standard operand

pos: + self
neg: - self
add: self + other
sub: self - other
radd: other + self
rsub: other - self
mul: self * other
rmul: other * self
truediv: self / other
rtruediv: other / self
pow: self**other
"""


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), '+')
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data-other.data, (self, other), '-')
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')
        return out

    def __truediv__(self, other):
        return self * other**-1

    def  __pow__(self, other):
        # assert isinstance(other, (int, float)), "Only int/float"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, (self,), f'**{other}')
        return out

    def __pos__(self): # + self
        return self

    def __neg__(self): # - self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return -(self-other)

    def __rmul__(self, other): # other * self
        return self * other

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'

