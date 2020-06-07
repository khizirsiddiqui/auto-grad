from autograd import derivative
from dualnumber import tan, cos, sin, exp, log

def _check_deriv(fn, x, g):
    h = 1.e-7
    ac = 1.e-5
    rc = 1.e-8

    _g = (fn(x + h) - fn(x))/h
    lim = rc + abs(_g) * ac
    return abs(g - _g) <= lim

def check_deriv(fn, x, args, g=None):
    if g is None:
        return _check_deriv(fn, x, args)

    h = 1.e-7
    ac = 1.e-5
    rc = 1.e-8

    fn_args = args[:]
    fn_args[x] += h

    _g = (fn(*fn_args) - fn(*args))/h
    lim = rc + abs(_g) * ac
    return abs(g - _g) <= lim

def test_tan():
    g = derivative(tan, 0.5)
    assert check_deriv(tan, 0.5, g)

def test_cos():
    g = derivative(cos, 0.5)
    assert check_deriv(cos, 0.5, g)

def test_log():
    g = derivative(log, 0.5)
    assert check_deriv(log, 0.5, g)

def test_sin():
    g = derivative(sin, 0.5)
    assert check_deriv(sin, 0.5, g)

def test_exp():
    g = derivative(exp, 0.5)
    assert check_deriv(exp, 0.5, g)

def test_fn_deriv():
    def f(x, y, z):
        return sin(y ** (x + 1)) - x * y * log((x ** 2) * 3)

    g = derivative(f, 1, [0.5, 4, -2.3])
    assert check_deriv(f, 1, [0.5, 4, -2.3], g)