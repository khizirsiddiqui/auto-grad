import graph.api as np

import numpy as np

def grad_add(node, prev_adjoint):
    return [prev_adjoint, prev_adjoint]

def grad_sub(node, prev_adjoint):
    return [prev_adjoint, -1 * prev_adjoint]

def grad_mul(node, prev_adjoint):
    return [
        prev_adjoint * node.op2,
        prev_adjoint * node.op1
    ]

def grad_div(node, prev_adjoint):
    return [
        prev_adjoint / node.op2,
        -1 * prev_adjoint * node.op1 / node.op2 ** 2
    ]

def grad_pow(node, prev_adjoint):
    return [
        prev_adjoint * node.op2 * (node.op1 ** (node.op2 - 1)),
        prev_adjoint * node * np.log(node.op1)
    ]

def grad_transpose(node, prev_adjoint):
    return [prev_adjoint.T, None]

def grad_sum(node, prev_adjoint):
    return [prev_adjoint * np.ones_like(node.op1), None]

def grad_mean(node, prev_adjoint):
    return [prev_adjoint * node * np.ones_like(node.op1), None]

def grad_exp(node, prev_adjoint):
    return [prev_adjoint * node, None]

def grad_log(node, prev_adjoint):
    return [prev_adjoint * (1. / node.op1), None]

def grad_max(node, prev_adjoint):
    dop1 = np.where(node.op1 == node.with_keepdims, 1, 0)
    normalizers = np.sum(dop1, axis=node.axis, keepdims=True)
    normalized_dop1 = dop1 / normalizers

    return [prev_adjoint * normalized_dop1, None]

def grad_dot(node, prev_adjoint):
    prev_adj = prev_adjoint
    op1 = node.op1
    op2 = node.op2

    if prev_adjoint.ndim * node.op2.ndim == 1:
        prev_adj = np.reshape(prev_adjoint, (-1, 1))
        op2 = np.reshape(op2, (-1, 1))
    if prev_adjoint.ndim * node.op1.ndim == 1:
        prev_adj = np.reshape(prev_adjoint, (-1, 1))
        op1 = np.reshape(op1, (-1, 1))
    return [
        np.dot(prev_adj, op2.T),
        np.dot(op1.T, prev_adj)
    ]

def grad_where(node, prev_adjoint):
    dop1 = np.zeros_like(node.op1)
    dop2 = np.ones_like(node.op2)

    dop1[node.condition] = 1
    dop2[node.condition] = 0

    return [prev_adjoint * dop1, prev_adjoint * dop2]

def grad_sin(node, prev_adjoint):
    return [prev_adjoint * np.cos(node.op1), None]

def grad_cos(node, prev_adjoint):
    return [-1 * prev_adjoint * np.sin(node.op1), None]

def grad_softmax_cross_entropy(node, prev_adjoint):
    return [
        prev_adjoint * (node.softmax - node.labels),
        None
    ]

def grad_reshape(node, prev_adjoint):
    return [
        np.reshape(prev_adjoint, node.op1.shape),
        None
    ]

def grad_squeeze(node, prev_adjoint):
    return [
        np.reshape(prev_adjoint, node.op1.shape),
        None
    ]
