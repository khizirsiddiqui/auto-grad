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