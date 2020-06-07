from dualnumber import DualNumber
import math

def test_add():
    x = DualNumber(4, 7)
    y = DualNumber(-1, 5)
    z = x + y
    assert z.a == 3
    assert z.b == 12

    z = x + 3
    assert z.a == 7
    assert z.b == 7

    z = 3 + x
    assert z.a == 7
    assert z.b == 7

def test_sub():
    x = DualNumber(4, 7)
    y = DualNumber(-1, 5)
    z = x - y
    assert z.a == 5
    assert z.b == 2

    z = x - 3
    assert z.a == 1
    assert z.b == 7

    z = 3 - x
    assert z.a == -1
    assert z.b == -7

def test_mul():
    x = DualNumber(4, 7)
    y = DualNumber(-1, 5)
    z = x * y
    assert z.a == -4
    assert z.b == 13

    z = x * 3
    assert z.a == 12
    assert z.b == 21

    z = 3 * x
    assert z.a == 12
    assert z.b == 21

def test_div():
    x = DualNumber(3, 12)
    y = DualNumber(-3, -3)
    z = x / y
    assert z.a == -1
    assert z.b == -3

    z = x / 3
    assert z.a == 1
    assert z.b == 4

    z = 3 / x
    assert z.a == 1
    assert z.b == -4

def test_pow():
    x = DualNumber(3, 1)
    y = DualNumber(1, 2)
    z = x ** y
    b = 6 * math.log(3) + 1
    assert z.a == 3
    assert z.b == b

    z = x ** 2
    assert z.a == 9
    assert z.b == 6

    z = 2 ** x
    b = 8 * math.log(2)
    assert z.a == 8
    assert z.b == b
