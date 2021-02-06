
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
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
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

    def backward(self):
        """Compute back-propagation."""
        # Build topological order
        topo = list()
        visited = set()
        def build_topo(v):
            """"""
            # If not in `visited` 
            # otherwise add previous node to topo
            print('check node', v)

            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        print('topo')
        for n in topo:
            print(n)

    def __pos__(self): # + self
        return self

    def __neg__(self): # - self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

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

import pprint as p
a = Value(-3.0)
b = Value(3.0)
c = a + b
d = 2 * c
e = d / b

x = Value(2)
y = Value(4)
z = x / y
print(z._prev)

# print('a', a._prev)
# print('b', b._prev)
# print('c', c._prev)
# print('d', d._prev)
# print('e', e._prev)

# e.backward()


