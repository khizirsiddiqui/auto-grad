import numpy as np
from collections import deque

class Node(np.ndarray):
    def __new__(
        subtype, shape, dtype=float, buffer=None, offset=0, strides=None,
        order=None
    ):

        return np.ndarray.__new__(subtype, shape, dtype, buffer, offset,
                                  strides, order)

    def _create_node(self, method, node, name, self_first=True):
        if not isinstance(node, Node):
            node = Constant.create(node)
        val = getattr(np.ndarray, method)(self, node)
        return Operation.create(name, val, self if self_first else other,
                                other if self_first else self)

    def __add__(self, node):
        return self._create_node('__add__', other, 'add')

    def __radd__(self, node):
        return self._create_node('__radd__', other, 'add')

    def __sub__(self, node):
        return self._create_node('__sub__', other, 'sub')

    def __rsub__(self, node):
        return self._create_node('__rsub__', other, 'sub', False)

    def __mul__(self, node):
        return self._create_node('__mul__', other, 'mul')

    def __rmul__(self, node):
        return self._create_node('__rmul__', other, 'mul', False)

    def __div__(self, node):
        return self._create_node('__div__', other, 'div')

    def __rdiv_(self, node):
        return self._create_node('__rdiv__', other, 'div', False)

    def __truediv__(self, node):
        return self._create_node('__truediv__', other, 'div')

    def __rtruediv__(self, node):
        return self._create_node('__rtruediv__', other, 'div', False)
    
    def __pow__(self, node):
        return self._create_node('__pow__', other, 'pow')
    
    def __rpow__(self, node):
        return self._create_node('__rpow__', other, 'pow', False)
    
    @property
    def T(self):
        val = np.transpose(self)
        return Operation.create('transpose', val, self)

class Operation(Node):

    unknown_nodes = {}

    @staticmethod
    def create(op_name, op_result, op1, op2=None, name=None):
        obj = Operation(strides=op_result.strides, shape=op_result.shape,
                     dtype=op_result.dtype, buffer=np.copy(op_result))
        obj.op_name = op_name
        obj.op1 = op1
        obj.op2 = op2

        if name is not None:
            obj.name = name
        else:
            if op_name not in Operation.unknown_nodes:
                Operation.unkown_nodes[op_name] = 0
            nodeID = Operation.unknown_nodes[op_name]
            Operation.unknown_nodes[op_name] += 1
            obj.name = op_name + '_' + str(nodeID)

        return obj


class Constant(Node):
    unknown_count = 0

    @staticmethod
    def create(val, name=None):
        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)
        obj = Constant(strides=val.strides, shape=val.shape, dtype=val.dtype,
                       buffer=np.copy(val))
        if name is None:
            obj.name = "const_" + str(Constant.unknown_count)
            Constant.unknown_count += 1
        else:
            obj.name = name

        return obj

class Variable(Node):
    unknown_count = 0

    @staticmethod
    def create(val, name=None):
        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)
        obj = Variable(strides=val.strides, shape=val.shape, dtype=val.dtype,
                       buffer=np.copy(val))
        if name is None:
            obj.name = "var_" + str(Constant.unknown_count)
            Constant.unknown_count += 1
        else:
            obj.name = name

        return obj

class NodeQueue:
    def __init__(self):
        self.nodes = deque()
        self.node_names = deque()

    def push(self, node):
        self.nodes.append(node)
        self.node_names.append(node.name)

    def pop(self, node):
        self.node_names.popleft()
        return self.nodes.popleft()

    def __contains__(self, node):
        return node.name in self.node_names

    def __len__(self):
        return len(self.node_names)
