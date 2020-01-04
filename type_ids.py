class VecType(object):
    def __init__(self, n):
        self.n = n
    def __eq__(self, other):
        return isinstance(other, VecType) and self.n == other.n
    def __str__(self):
        return "R^" + str(self.n)
    def __hash__(self):
        return self.n
    def can_apply_to(self, other):
        return False

class FuncType(object):
    def __init__(self, arg_type, ret_type):
        self.arg_type = arg_type
        self.ret_type = ret_type
    def __eq__(self, other):
        return (isinstance(other, FuncType) and 
               self.arg_type == other.arg_type and
               self.ret_type == other.ret_type)
    def can_apply_to(self, other):
        return self.arg_type == other
    def __str__(self):
        paren_arg = False
        if (isinstance(self.arg_type, FuncType)):
            paren_arg = True
        arg_part = self.arg_type.__str__()
        if (paren_arg):
            arg_part = "(" + arg_part + ")"
        return arg_part + "->" + self.ret_type.__str__()
    def __hash__(self):
        return hash(self.arg_type) * 31 + hash(self.ret_type)
