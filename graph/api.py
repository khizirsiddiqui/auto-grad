from .graph import Node, Variable, Operation, Constant

import numpy as np


def variable(val, name=None):
    return Variable.create(val, name)

def constant(val, name=None):
    return Constant.create(val, name)

def sum(array, axis=None, keepdims=False, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.sum(array, axis=axis, keepdims=keepdims)
    return Operation.create('sum', val, array, name=name)

def mean(array, axis=None, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.mean(array, axis=axis)
    return Operation.create('mean', val, array, name=name)

def exp(array, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.exp(array)
    return Operation.create('exp', val, array, name=name)

def log(array, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.log(array)
    return Operation.create('log', val, array, name=name)

def max(array, axis=None, keepdims=False, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.max(array, axis=axis, keepdims=keepdims)
    return Operation.create('max', val, array, name=name)

def min(array, axis=None, keepdims=False, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.min(array, axis=axis, keepdims=keepdims)
    return Operation.create('min', val, array, name=name)

def dot(a, b, name=None):
    if not isinstance(a, Node):
        a = Constant.create(a)
    if not isinstance(b, Node):
        b = Constant.create(b)
    val = np.dot(a, b)
    return Operation.create('dot', val, a, b, name=name)

def _where(condition, a, b, opname, name=None):
    if not isinstance(a, Node):
        a = np.full_like(condition, a)
        a = Constant.create(a)
    if not isinstance(b, Node):
        b = np.full_like(condition, b)
        b = Constant.create(b)
    val = np.where(condition, a, b)
    op = Operation.create(opname, val, a, b, name=name)
    op.condition = condition
    return op

def where(condition, a, b, name=None):
    return _where(condition, a, b, 'where', name=name)

def relu(array, name='relu'):
    return _where(array > 0, array, 0, 'relu', name=name)

def sin(array, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.sin(array)
    return Operation.create('sin', val, array, name=name)

def cos(array, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.cos(array)
    return Operation.create('cos', val, array, name=name)

def tan(array, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.tan(array)
    return Operation.create('tan', val, array, name=name)

def reshape(array, shape, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.reshape(array, shape)
    return Operation.create('reshape', val, array, name=name)

def squeeze(array, axis=None, name=None):
    if not isinstance(array, Node):
        array = Constant.create(array)
    val = np.squeeze(array, axis=axis)
    return Operation.create('squeeze', val, array, name=name)

def softmax_cross_entropy(logits, y, name=None):
    if not isinstance(logits, Node):
        logits = Constant.create(logits)
    if not isinstance(y, Node):
        y = Constant.create(y)

    e_op = np.exp(logits - np.max(logits, axis=1, keepdims=1))
    softmax = e_op / np.sum(e_op, axis=1, keepdims=True)
    val = -1 * np.mean(y * np.log(softmax + 1e-6))

    op = Operation.create('softmax_cross_entropy', val, logits, name=name)
    op.softmax = softmax
    op.labels = y
    return op

def sanitize():
    Operation.unknown_nodes = {}
    Constant.unknown_count = 0
    Variable.unknown_count = 0

