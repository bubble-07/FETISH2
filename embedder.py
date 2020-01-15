import numpy as np
import scipy as sp
import random
from type_ids import *
from toposort import toposort_flatten
from func_embedding import get_embedding_matrix
import function_step
from interpreter_support import *
from vptree import VpTree

#Gets the diameter of a given embedding matrix
def diameter(embed_matrix):
    dist_mat = sp.spatial.distance_matrix(embed_matrix, embed_matrix)
    #Return the str8 up max distance
    return np.amax(dist_mat)



#Responsible for maintaining information about all function
#embeddings based on some associated interpreter state
class EmbedderState(object):
    def __init__(self, interpreter_state):
        self.interpreter_state = interpreter_state
        #Dictionary from types to matrices of terms' embeddings
        self.embeddings = {}
        #Dictionary from type, function term index to vantage point tree for known func args 
        self.vptrees = {}
        #Dictionary from function type to a list of arg index -> ret index maps
        #Intended to be used for "target functions"
        self.target_functions = {}

        self.target_embeddings = {}

    #Given a function type and a mapping from arg indices to ret indices,
    #add it to our target functions list [will be considered in the generation of embeddings,
    #but will not pollute extra garbo in actual application tables]
    def add_target(self, func_type, in_out_mapping):
        if (not (func_type in self.target_functions)):
            self.target_functions[func_type] = []
        self.target_functions[func_type].append(in_out_mapping)

    #Ensures that all functions of the same type as functions
    #which are specified as targets are evaluated at all of the same
    #points that the target is defined on
    def ensure_all_targets_evaluated(self):
        for func_type in self.target_functions:
            arg_type = func_type.arg_type
            arg_inds = set([])
            for func in self.target_functions[func_type]:
                for arg_ind in func:
                    arg_inds.add(arg_ind)
            arg_space = self.interpreter_state.get_type_space(arg_type)
            func_space = self.interpreter_state.get_type_space(func_type)
            for func_ind in range(len(func_space.terms)):
                func_ptr = TermPointer(func_space, func_ind)
                for arg_ind in arg_inds:
                    arg_ptr = TermPointer(arg_space, arg_ind)
                    self.interpreter_state.apply_ptrs(func_ptr, arg_ptr)

    def refresh(self):
        self.embeddings = {}
        self.target_embeddings = {}
        self.vptrees = {}
        self.init_embeddings()

    def __str__(self):
        result = "Embeddings: \n"
        for kind in self.embeddings:
            result += str(kind) + ":\n"
            result += str(self.embeddings[kind]) + "\n"
        return result

    def embed_term(self, term, kind):
        #Easy case: vectors embed to themselves
        if isinstance(kind, VecType):
            return term.get()
        #Otherwise, we need to use a lookup
        embedding = self.embeddings[kind]
        #Get the index of the term in its corresponding type space
        space = self.interpreter_state.type_spaces[kind]
        ind = space.get_pointer(term).get_index()
        return embedding[ind]

    def embed_ptr(self, term_ptr):
        return self.embed_index(term_ptr.get_index(), term_ptr.get_type())

    def embed_index(self, ind, kind):
        if isinstance(kind, VecType):
            space = self.interpreter_state.type_spaces[kind]
            return space.get(ind).get()
        embedding = self.embeddings[kind]
        return embedding[ind]

    def embed_target_index(self, target_ind, kind):
        embedding_mat = self.target_embeddings[kind]
        return embedding_mat[target_ind]

    #Gets the full matrix of embeddings for a given kind
    def get_embed_matrix(self, kind):
        if isinstance(kind, VecType):
            space = self.interpreter_state.type_spaces[kind]
            result = []
            for ind in range(len(space.terms)):
                result.append(space.get(ind).get())
            return np.vstack(result)
        return self.embeddings[kind]

    #Gets the diameter of the embedding space for the given kind
    def get_diameter(self, kind):
        return diameter(self.get_embed_matrix(kind))

    #Get a pointer to the term of a given kind which is closest to the specified
    #embedding point
    def get_closest_term_ptr(self, kind, target_vector):
        embed_matrix = self.get_embed_matrix(kind)
        dists = embed_matrix - target_vector.reshape((1, -1))
        dists = np.linalg.norm(dists, axis=1)
        ind = np.argmin(dists)
        space = self.interpreter_state.type_spaces[kind]
        return dists[ind], TermPointer(space, ind)

    #Given a function type T -> R, and the embedding of some desired
    #result, returns the TermApplication of a T->R to a T which gets closest
    #among all currently-memoized application results
    def get_closest_term_application(self, func_type, ret_target):
        arg_type = func_type.arg_type
        ret_type = func_type.ret_type
        #Keep a copy of the embedding matrix for the ret type
        #sitting around here, because we'll need it. Also compute
        #all distances to the target, for convenience
        embed_matrix = self.get_embed_matrix(ret_type)
        dists = embed_matrix - ret_target.reshape((1, -1))
        dists = np.linalg.norm(dists, axis=1)

        min_func_ind = None
        min_arg_ind = None
        min_ret_dist = float('+inf')

        table = self.interpreter_state.get_application_table(func_type)
        for func_ind, arg_ind, ret_ind in table.get_index_triple_generator():
            dist = dists[ret_ind]
            if (dist == min_ret_dist):
                #Here, we need to flip a coin, otherwise, we won't properly
                #wind up exploring the state space if many of the application pairs
                #have the same output
                #We don't care how skewed this may be, which is why it be how it do
                if (random.random() > 0.5):
                    min_func_ind = func_ind
                    min_arg_ind = arg_ind
                    min_ret_dist = dist

            if (dist < min_ret_dist):
                min_func_ind = func_ind
                min_arg_ind = arg_ind
                min_ret_dist = dist
        func_space = self.interpreter_state.type_spaces[func_type]
        arg_space = self.interpreter_state.type_spaces[arg_type]
        func_ptr = TermPointer(func_space, min_func_ind)
        arg_ptr = TermPointer(arg_space, min_arg_ind)
        return min_ret_dist, TermApplication(func_ptr, arg_ptr)

    #Given a function pointer, returns a pairing of matrices of shapes
    #m x d and m x d_two representing all computed input/output embedding pairs
    def get_embedding_along_arg_dimension(self, func_type, func_ptr):
        func_index = func_ptr.get_index()
        arg_type = func_type.arg_type
        ret_type = func_type.ret_type
        arg_embeds = self.get_embed_matrix(arg_type)
        ret_embeds = self.get_embed_matrix(ret_type)
        func_table = self.interpreter_state.application_tables[func_type]

        inputs = []
        outputs = []
        func_row = func_table.table[func_index]
        for arg_index in func_row:
            ret_index = func_row[arg_index]
            inputs.append(arg_embeds[arg_index])
            outputs.append(ret_embeds[ret_index])
        inputs = np.vstack(inputs)
        outputs = np.vstack(outputs)
        return inputs, outputs

    #Like the previous method, but along function space instead.
    #For this, we need to do a bit more work than the other case,
    #because if we're not careful, we could miss entirely. Hence the need
    #for interpolation
    def get_embedding_along_func_dimension(self, func_type, arg_ptr):
        func_vptrees = self.vptrees[func_type]
        arg_type = func_type.arg_type
        ret_type = func_type.ret_type
        func_space = self.interpreter_state.type_spaces[func_type]

        arg_embedding = self.embed_index(arg_ptr.get_index(), arg_type)

        func_pairs = []
        for func_ind in range(len(func_space.terms)):
            func_ptr = TermPointer(func_space, func_ind)
            embed_pair = self.get_embedding_along_arg_dimension(func_type, func_ptr)
            func_pairs.append(embed_pair)
        outputs = function_step.impute_input_slice(arg_embedding, func_pairs, func_vptrees)

        #Great, now we just need to compile the inputs (which are functions, lel)
        #Since we attach targets, and those show up in the embedding matrix, we need this len thing
        inputs = self.get_embed_matrix(func_type)[:len(func_space.terms)]

        return inputs, outputs

    #Gets the (inputs, outputs) pairing for a given function row
    def get_func_pair(self, arg_type, ret_type, func_row):
        arg_embeds = []
        ret_embeds = []
        for arg_ind in func_row:
            ret_ind = func_row[arg_ind]
            arg_embed = self.embed_index(arg_ind, arg_type)
            ret_embed = self.embed_index(ret_ind, ret_type)
            arg_embeds.append(arg_embed)
            ret_embeds.append(ret_embed)
        arg_embeds = np.vstack(arg_embeds)
        ret_embeds = np.vstack(ret_embeds)
        return (arg_embeds, ret_embeds)


    def init_embeddings(self):
        #To construct embeddings, we need to topologically sort
        #all types
        #Construct the dictionary of dependencies
        depends = {}
        for kind in self.interpreter_state.application_tables:
            depends[kind] = {kind.arg_type, kind.ret_type}

        #Contains types in dependency order
        sorted_types = toposort_flatten(depends, sort=False)

        for kind in sorted_types:
            if isinstance(kind, VecType):
                #Don't need to derive embeddings for vector types
                continue
            #Instantiate the vp tree dict for the kind
            self.vptrees[kind] = {}

            #Derive the embedding for the type 
            arg_type = kind.arg_type
            ret_type = kind.ret_type
            space = self.interpreter_state.get_type_space(kind)
            table = self.interpreter_state.get_application_table(kind)
            func_pairs = []
            for func_ind in range(len(space.terms)):
                func_row = table.table[func_ind]
                func_pair = self.get_func_pair(arg_type, ret_type, func_row)
                func_pairs.append(func_pair)
                #Func pairs were updated, but we also need to compute the vantage point
                #tree for all of the arguments in arg_embeds
                arg_embeds, _ = func_pair
                tree = VpTree(arg_embeds)
                self.vptrees[kind][func_ind] = tree
            #Not done yet -- we need to also consider the target functions
            if (kind in self.target_functions):
                func_ind = len(space.terms)
                target_map = self.target_functions[kind]
                for in_out_map in target_map:
                    func_pair = self.get_func_pair(arg_type, ret_type, in_out_map)
                    func_pairs.append(func_pair)

                    arg_embeds, _ = func_pair
                    tree = VpTree(arg_embeds)
                    self.vptrees[kind][func_ind] = tree

                    func_ind += 1

            #Now that we have the embedded input, output pairs, 
            #we can get the embedding matrix for this type space
            X = self.get_embed_matrix(arg_type) #Used for convex hull evaluation
            X = np.transpose(X)
            embedding_matrix = get_embedding_matrix(func_pairs, X, self.vptrees[kind])
            diam = diameter(embedding_matrix)
            if (diam != 0.0):
                embedding_matrix = embedding_matrix / diameter(embedding_matrix)
            #Not done yet -- need to split it out to embeddings and target_embeddings
            if (len(space.terms) < embedding_matrix.shape[0]):
                target_embedding_matrix = embedding_matrix[len(space.terms):]
                self.target_embeddings[kind] = target_embedding_matrix
            embedding_matrix = embedding_matrix[:len(space.terms)]
            self.embeddings[kind] = embedding_matrix
