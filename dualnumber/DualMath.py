from .DualNumber import DualNumber
import math


def sin(x):
    if isinstance(x, DualNumber):
        return DualNumber(math.sin(x.a), math.cos(x.a) * x.b)
    else:
        return math.sin(x)

def cos(x):
    if isinstance(x, DualNumber):
        return DualNumber(math.cos(x.a), -1 * math.sin(x.a) * x.b)
    else:
        return math.cos(x)

def tan(x):
    if isinstance(x, DualNumber):
        return DualNumber(math.tan(x.a), x.b / (math.cos(x.a)**2))
    else:
        return math.tan(x)

def log(x, base=math.e):
    if isinstance(x, DualNumber):
        a = math.log(x.a, base)
        b = x.b / (math.log(base) * x.a)
        return DualNumber(a, b)
    else:
        return math.log(x, base)

def exp(x):
    if isinstance(x, DualNumber):
        val = math.exp(x.a)
        return DualNumber(val, val * x.b)
    else:
        return math.exp(x)

def pow(x, num):
    return x ** num
