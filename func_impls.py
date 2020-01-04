import numpy as np
from type_ids import VecType, FuncType
from basic_terms import *

class FuncImpl(object):
    def __init__(self):
        pass
    def ret_type(self):
        raise NotImplemented
    def required_arg_types(self):
        raise NotImplemented
    def ready_to_evaluate(self, partial_args):
        return len(partial_args) == len(self.required_arg_types())
    def evaluate(self, interpreter_state, args):
        raise NotImplemented

class MapImpl(FuncImpl):
    def __init__(self, n):
        self.n = n
    def required_arg_types(self):
        return [FuncType(VecType(1), VecType(1)), VecType(self.n)]
    def ret_type(self):
        return VecType(self.n)
    def __eq__(self, other):
        return (isinstance(other, MapImpl) and
                self.n == other.n)
    def __hash__(self):
        return self.n
    def __str__(self):
        return "Map[" + str(self.n) + "]"
    def evaluate(self, interpreter_state, args):
        func_term = args[0]
        func_type = FuncType(VecType(1), VecType(1))
        arg_term = args[1].get()
        result = np.zeros(self.n)
        for i in range(self.n):
            arg_val = VectorTerm(np.array([arg_term[i]]))
            result_term = interpreter_state.blind_apply(func_term, func_type,
                                          arg_val, VecType(1))
            result[i] = result_term.get()[0]
        return VectorTerm(result)

class ComposeImpl(FuncImpl):
    def __init__(self, n, m, p):
        self.n = n
        self.m = m
        self.p = p
    def required_arg_types(self):
        return [FuncType(VecType(self.m), VecType(self.p)), 
                FuncType(VecType(self.n), VecType(self.m)), 
                VecType(self.n)]
    def ret_type(self):
        return FuncType(self.n, self.p)
    def __eq__(self, other):
        return (isinstance(other, ComposeImpl) and
                self.n == other.n and
                self.m == other.m and
                self.p == other.p)
    def __hash__(self):
        return 31 * 31 * self.n + 31 * self.m + self.p
    def __str__(self):
        return "Compose[" + str(self.n) + "," + str(self.m) + "," + str(self.p) + "]"
    def evaluate(self, interpreter_state, args):
        first_func = args[1]
        first_func_type = FuncType(VecType(self.n), VecType(self.m))
        second_func = args[0]
        second_func_type = FuncType(VecType(self.m), VecType(self.p))
        arg_value = args[2]
        arg_type = VecType(self.n)

        middle_value = interpreter_state.blind_apply(first_func, first_func_type,
                                         arg_value, arg_type)
        middle_type = VecType(self.m)
        return interpreter_state.blind_apply(second_func, second_func_type,
                                         middle_value, middle_type)
       
class FillImpl(FuncImpl):
    def __init__(self, n):
        self.n = n
    def required_arg_types(self):
        return [VecType(1)]
    def ret_type(self):
        return VecType(self.n)
    def __eq__(self, other):
        return (isinstance(other, FillImpl) and self.n == other.n)
    def __hash__(self):
        return self.n
    def __str__(self):
        return "Fill[" + str(self.n) + "]"
    def evaluate(self, interpreter_state, args):
        arg_value = args[0].get()[0]
        return VectorTerm(np.full(self.n, arg_value))

class ReduceImpl(FuncImpl):
    def __init__(self, n):
        self.n = n
    def required_arg_types(self):
        return [FuncType(VecType(1), FuncType(VecType(1), VecType(1))),
                VecType(1), VecType(self.n)]
    def ret_type(self):
        return VecType(1)
    def __eq__(self, other):
        return isinstance(other, ReduceImpl) and self.n == other.n
    def __hash__(self):
        return self.n
    def __str__(self):
        return "Reduce[" + str(self.n) + "]"
    def evaluate(self, interpreter_state, args):
        func_val = args[0]
        accum_val = args[1]
        list_args = args[2].get()

        unary_func_type = FuncType(VecType(1), VecType(1))
        binary_func_type = FuncType(VecType(1), unary_func_type)
        for i in range(self.n):
            arg_val = VectorTerm(np.array([list_args[i]]))
            curry_man = interpreter_state.blind_apply(func_val, binary_func_type,
                                   arg_val, VecType(1)) 
            accum_val = interpreter_state.blind_apply(curry_man, unary_func_type,
                                   accum_val, VecType(1))
        return accum_val

class BinaryFuncImpl(FuncImpl):
    def __init__(self, n):
        self.n = n
    def required_arg_types(self):
        return [VecType(self.n), VecType(self.n)]
    def ret_type(self):
        return VecType(self.n)
    def __hash__(self):
        return self.n
    def __eq__(self, other):
        return type(self) == type(other) and self.n == other.n
    def short_name(self):
        raise NotImplemented
    def __str__(self):
        return self.short_name() + "[" + str(self.n) + "]"
    def evaluate(self, interpreter_state, args):
        return VectorTerm(self.impl(args[0].get(), args[1].get()))
    def impl(self, x, y):
        raise NotImplemented

class AddImpl(BinaryFuncImpl):
    def __init__(self, n):
        super(AddImpl, self).__init__(n)
    def impl(self, x, y):
        return x + y
    def short_name(self):
        return "+"

class SubImpl(BinaryFuncImpl):
    def __init__(self, n):
        super(SubImpl, self).__init__(n)
    def impl(self, x, y):
        return x - y
    def short_name(self):
        return "-"

class MultImpl(BinaryFuncImpl):
    def __init__(self, n):
        super(MultImpl, self).__init__(n)
    def impl(self, x, y):
        return x * y
    def short_name(self):
        return "*"

class DivImpl(BinaryFuncImpl):
    def __init__(self, n):
        super(DivImpl, self).__init__(n)
    def impl(self, x, y):
        result = x / y
        result[np.isinf(result)] = 0.0
        result[np.isnan(result)] = 0.0
        return result
    def short_name(self):
        return "/"
