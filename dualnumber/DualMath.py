from .DualNumber import DualNumber
import math


def sin(x):
    if isinstance(x, DualNumber):
        return DualNumber(math.sin(x.a), math.cos(x.a) * x.b)
    else
        return math.sin(x)

def cos(x):
    if isinstance(x, DualNumber):
        return DualNumber(math.cos(x.a), -1 * math.sin(x.a) * x.b)
    else
        return math.cos(x)

def tan(x):
    if isinstance(x, DualNumber):
        return DualNumber(math.tan(x.a), (math.sec(x.a)**2) * x.b)
    else
        return math.tan(x)

def log(x, base=math.e):
    if isinstance(x, DualNumber):
        return DualNumber(math.log(x.a, base), x.b / (math.log(base) * x.a))
    else
        return math.log(x, base=base)

def exp(x):
    if isinstance(x, DualNumber):
        val = math.exp(x.a)
        return DualNumber(val, val)
    else
        return math.tan(x)

def pow(x, num):
    return x ** num
