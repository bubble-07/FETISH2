import numpy as np

#Reference to a given typespace, index pair
class TermPointer(object):
    def __init__(self, type_space, index):
        self.type_space = type_space
        self.index = index
    def get_index(self):
        return self.index
    def get_term(self):
        return self.type_space.get(self.index)
    def get_type(self):
        return self.type_space.my_type
    def __str__(self):
        return "Term(" + str(self.index)  + "," + str(self.type_space.my_type) + ")"
    def __eq__(self, other):
        return (isinstance(other, TermPointer) and 
                self.type_space == other.type_space and
                self.index == other.index)
    def __hash__(self):
        return hash(self.type_space) * 31 + hash(self.index)

#(pseudo-)Term which is composed of the application of one term to another
#Strictly speaking, these should never be stored in any of the type spaces,
#since they are actually not fully evaluated terms
#You probably want "PartiallyAppliedTerm"
class TermApplication(object):
    def __init__(self, func_ptr, arg_ptr):
        func_type = func_ptr.get_type()
        arg_type = arg_ptr.get_type()
        if (not (func_type.can_apply_to(arg_type))):
            raise ValueError("Cannot apply " + str(func_type) + " to " + str(arg_type))
        self.func_ptr = func_ptr
        self.arg_ptr = arg_ptr
    def __eq__(self, other):
        return (isinstance(other, TermApplication) and
                self.func_ptr == other.func_ptr and
                self.arg_ptr == other.arg_ptr)
    def __hash__(self):
        return 31 * hash(self.func_ptr) + hash(self.arg_ptr)
    def __str__(self):
        return "[" + str(self.func_ptr) + " " + str(self.arg_ptr) + "]"

