from value import Value
from pprint import pprint
import random

a = Value(2.0)
b = Value(4.0)
c = a + b
d = c ** 2
e = d / b 
e.backward()

print(a.grad) # dz / da
print(b.grad) # dz / db
#print(c.grad)




