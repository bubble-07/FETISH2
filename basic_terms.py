import numpy as np

#Represents a (scalar or vector) term
class VectorTerm(object):
    def __init__(self, numpy_array):
        self.numpy_array = numpy_array
    def __str__(self):
        return str(self.numpy_array)
    def __eq__(self, other):
        return (isinstance(other, VectorTerm) and
                np.array_equal(self.numpy_array, other.numpy_array))
    def get(self):
        return self.numpy_array
    def __hash__(self):
        return hash(self.numpy_array.sum())

#Represents a term which has been partially applied to
#a collection of arguments, but does not yet have enough arguments
#to evaluate the corresponding FuncImpl
class PartiallyAppliedTerm(object):
    def __init__(self, impl, partial_args):
        self.impl = impl
        self.partial_args = tuple(partial_args)
    def __eq__(self, other):
        return (isinstance(other, PartiallyAppliedTerm) and
                self.impl == other.impl and
                self.partial_args == other.partial_args)
    def __hash__(self):
        return hash(self.impl) + 31 * hash(self.partial_args)
    def __str__(self):
        result = str(self.impl) + "(" + str(self.partial_args[0])
        for i in range(1, len(self.partial_args)):
            arg = self.partial_args[i]
            result += ", " + str(arg)
        result += ")"
        return result
