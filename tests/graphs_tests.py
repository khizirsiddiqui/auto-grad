import numpy as np
from graph.graph import Constant, Variable
import graph as g


def test_add():
    x = Constant.create(np.ones(2), 'x')
    y = Variable.create(np.ones(2) * 3, 'y')
    z = x + y
    assert (z == np.ones(2) * 4).all()

def test_api():
    x = g.constant(2)
    y = g.constant(3)
    z = x + y
    assert z == 5
    
    z = x ** y
    assert z == 8

def test_grad():
    x = g.variable(2, name='x')
    y = g.constant(3)
    z = x + y
    grad = z.backward()
    assert len(grad) == 1
    assert grad['x'] == 1

    z = x**3 + y**2
    grad = z.backward()
    assert len(grad) == 1
    assert grad['x'] == 12

    y = g.variable(1, name='y')
    z = g.sin(2 * x + y)
    grad = z.backward()
    assert len(grad) == 2
    assert grad['x'] == 2 * g.cos(2 * x + y)
    assert grad['y'] == g.cos(2 * x + y)
