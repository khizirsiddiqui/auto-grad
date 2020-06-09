import numpy as np


class Node(np.ndarray):
    def __new__(
        subtype, shape, dtype=float, buffer=None, offset=0, strides=None,
        order=None
    ):

        return np.ndarray.__new__(subtype, shape, dtype, buffer, offset,
                                  strides, order)


class OpNode(Node):

    unknown_nodes = {}

    @staticmethod
    def create(op_name, op_result, op1, op2=None, name=None):
        obj = OpNode(strides=op_result.strides, shape=op_result.shape,
                     dtype=op_result.dtype, buffer=np.copy(op_result))
        obj.op_name = op_name
        obj.op1 = op1
        obj.op2 = op2

        if name is not None:
            obj.name = name
        else:
            if op_name not in OpNode.unknown_nodes:
                OpNode.unkown_nodes[op_name] = 0
            nodeID = OpNode.unknown_nodes[op_name]
            OpNode.unknown_nodes[op_name] += 1
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

