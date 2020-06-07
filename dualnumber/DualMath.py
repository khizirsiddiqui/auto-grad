from numbers import Number
from .DualNumber import DualNumber
import math

def get_clean(x):
    if isinstance(x, Number):
        x = DualNumber(x, 0)
    return x

def sin(x):
    x = get_clean(x)
    return DualNumber(math.sin(x.a), math.cos(x.a) * x.b)

def cos(x):
    x = get_clean(x)
    return DualNumber(math.cos(x.a), -1 * math.sin(x.a) * x.b)

def tan(x):
    x = get_clean(x)
    return DualNumber(math.tan(x.a), 1/(math.cos(x.a)**2) * x.b)

def log(x, base=math.e):
    x = get_clean(x)
    return DualNumber(math.log(x.a, base), x.b / (math.log(base) * x.a))

def exp(x):
    x = get_clean(x)
    val = math.exp(x.a)
    return DualNumber(val, val)

def pow(x, num):
    x = get_clean(x)
    return x ** num
