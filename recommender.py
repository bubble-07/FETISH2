import numpy as np
import scipy as sp
import random
from type_ids import *
from basic_terms import *
from interpreter_support import *

#Inductively defined binary tree whose leaves are term pointers
#or numpy vectors [for constants which have yet to be added]
#Unlike a TermApplication, which is just a pairing of term pointers.
class SyntaxTree(object):
    def __init__(self, func_subtree, arg_subtree):
        self.func_subtree = func_subtree
        self.arg_subtree = arg_subtree
    def __eq__(self, other):
        return (isinstance(other, SyntaxTree) and
                self.func_subtree == other.func_subtree and
                self.arg_subtree == other.arg_subtree)
    def __hash__(self):
        return 31 * hash(self.func_subtree) + hash(self.arg_subtree)
    def __str__(self):
        func_string = str(self.func_subtree)
        arg_string = str(self.arg_subtree)
        
        if (isinstance(self.func_subtree, TermPointer)):
            func_string = str(self.func_subtree.get_term())
        if (isinstance(self.arg_subtree, TermPointer)):
            arg_string = str(self.arg_subtree.get_term())

        return "(" + func_string + " " + arg_string + ")"
    def evaluate(self, interpreter_state):
        func_eval = self.func_subtree
        if (isinstance(func_eval, SyntaxTree)):
            func_eval = func_eval.evaluate(interpreter_state)
        arg_eval = self.arg_subtree
        if (isinstance(arg_eval, SyntaxTree)):
            arg_eval = arg_eval.evaluate(interpreter_state)
        if (isinstance(arg_eval, np.ndarray)):
            #Gotta embed a new constant
            arg_type = VecType(arg_eval.shape[0])
            arg_space = interpreter_state.type_spaces[arg_type]
            arg_eval = arg_space.get_pointer(VectorTerm(arg_eval))
        #Both func_eval and arg_eval now contain pointers, so eval those
        result_ptr = interpreter_state.apply_ptrs(func_eval, arg_eval)
        return result_ptr
