import numpy as np
import scipy as sp
import random
from type_ids import *
from basic_terms import *
import idw_sample
import secant
from interpreter_support import *

#Exponent to use for idw in randomized "pull-from" table choice
TABLE_CHOICE_EXPONENT=1.0

#Exponent to use for idw in randomized "should continue?" choice
CONTINUE_CHOICE_EXPONENT=1.0

#Multiplier which determines how much continuation we seek
CONTINUE_MULTIPLIER=1.0

#Exponent to use for idw in randomized arg/function direction choice
ARG_FUNC_EXPONENT=0.1

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

class RecommenderState(object):
    def __init__(self, embedder_state):
        self.embedder_state = embedder_state
        self.interpreter_state = embedder_state.interpreter_state

    #Given the kind and index of a target attached to the embedder_state here,
    #perform one complete suggestion+update step
    def step(self, target_kind, target_ind):
        print "ensuring targets evaled"
        self.embedder_state.ensure_all_targets_evaluated()
        print "ensuring application rows filled"
        self.interpreter_state.ensure_every_application_table_row_filled()
        print "refreshing embedder state"
        self.embedder_state.refresh()
        target_embed = self.embedder_state.embed_target_index(target_ind, target_kind)
        print "getting a suggestion"
        tree = self.get_suggestion(target_kind, target_embed, force_continue=True)
        print "Suggestion: " + str(tree)
        print "evaluating tree"
        result_ptr = tree.evaluate(self.interpreter_state)
        return result_ptr
     
    #Given the type of a target element, and the embedding of the target,
    #perform one optimization iteration to try to return a SyntaxTree
    #of a term which is closer to the target
    def get_suggestion(self, target_type, target_embedding, force_continue=False):
        #Special case: if the target type is a vector type, then we know exactly what to suggest
        if (isinstance(target_type, VecType)):
            return target_embedding

        #First, pick the function application table we'd use to pull a term from [if we're going to do that]
        candidate_func_types = list(self.interpreter_state.get_funcs_returning(target_type))
        #Special case: if we have no candidate func types, return the closest term
        if (len(candidate_func_types) == 0):
            _, term_ptr = self.embedder_state.get_closest_term_ptr(target_type, target_embedding)
            return term_ptr

        dists = []
        term_apps = []
        for i in range(len(candidate_func_types)):
            func_type = candidate_func_types[i]
            dist, term_app = self.embedder_state.get_closest_term_application(func_type, target_embedding)
            dists.append(dist)
            term_apps.append(term_app)
        dists = np.array(dists)
        chosen_ind = idw_sample.idw_sample_from_dists(dists, TABLE_CHOICE_EXPONENT)
        func_type = candidate_func_types[chosen_ind]
        arg_type = func_type.arg_type
        term_app = term_apps[chosen_ind]
        term_app_dist = dists[chosen_ind]

        #Now, decide if we actually should continue to derive terms from applications
        #by comparing the chosen closest term application to all available terms
        term_dist, term_ptr = self.embedder_state.get_closest_term_ptr(target_type, target_embedding)

        print "Closest term: ", term_ptr.get_term()

        dists = np.array([term_dist * CONTINUE_MULTIPLIER, term_app_dist])
        continue_choice = idw_sample.idw_sample_from_dists(dists, CONTINUE_CHOICE_EXPONENT)
        should_continue = (continue_choice == 1) or force_continue
        if (should_continue):
            #We got the green light to try to find things that are closer
            #using the given term application as a starting point
            func_ptr = term_app.func_ptr
            arg_ptr = term_app.arg_ptr
            func_embed = self.embedder_state.embed_ptr(func_ptr)
            arg_embed = self.embedder_state.embed_ptr(arg_ptr)

            #Compute estimates of the effects of coordinate-descent-like
            #direction choice between choosing to modify the function
            #and choosing to modify the argument
            arg_ins, arg_outs = self.embedder_state.get_embedding_along_arg_dimension(func_type, func_ptr)
            funcs_in, funcs_out = self.embedder_state.get_embedding_along_func_dimension(func_type, arg_ptr)

            proposed_arg = secant.get_new_input_to_try_linear(arg_ins, arg_outs, target_embedding)
            #Before getting the proposed function, do a cardinality check. If there's only one function,
            #then we should just use the proposed arg modification
            if (funcs_in.shape[0] < 2):
                new_arg_term = self.get_suggestion(arg_type, proposed_arg)
                new_application = SyntaxTree(func_ptr, new_arg_term)
                return new_application

            proposed_func = secant.get_new_input_to_try_linear(funcs_in, funcs_out, target_embedding)

            func_diameter = self.embedder_state.get_diameter(func_type)
            arg_diameter = self.embedder_state.get_diameter(arg_type)

            arg_dist = np.linalg.norm(proposed_arg - arg_embed)
            func_dist = np.linalg.norm(proposed_func - func_embed)

            #We normalize because the two are by default inconsummeasurable in general
            arg_dist = arg_dist / arg_diameter
            func_dist = func_dist / func_diameter

            #Now, pick between the arg update and the function update
            #based on an inverse-distance-weighting of how far we'd update in each direction
            dists = np.array([arg_dist, func_dist])
            arg_func_choice = idw_sample.idw_sample_from_dists(dists, ARG_FUNC_EXPONENT)
            choice = "arg"
            if (arg_func_choice == 1):
                choice = "func"

            if (choice == "arg"):
                new_arg_term = self.get_suggestion(arg_type, proposed_arg)
                new_application = SyntaxTree(func_ptr, new_arg_term)
            if (choice == "func"):
                new_func_term = self.get_suggestion(func_type, proposed_func)
                new_application = SyntaxTree(new_func_term, arg_ptr)

            #Great, now we just need to return our new syntax tree
            return new_application

        else:
            #We're just giving up and yielding the closest term we already know about
            return term_ptr
            


