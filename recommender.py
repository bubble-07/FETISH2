import numpy as np
from type_ids import *
from toposort import toposort_flatten
from func_embedding import get_embedding_matrix

#Contains the actual meat of this thing: the recommendation
#engine which determines which expressions to evaluate

#Responsible for maintaining information about all function
#embeddings based on some associated interpreter state
class RecommenderState(object):
    def __init__(self, interpreter_state):
        self.interpreter_state = interpreter_state
        self.embeddings = {}
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

    def embed_index(self, ind, kind):
        if isinstance(kind, VecType):
            space = self.interpreter_state.type_spaces[kind]
            return space.get(ind).get()
        embedding = self.embeddings[kind]
        return embedding[ind]

    #Gets the full matrix of embeddings for a given kind
    def get_embed_matrix(self, kind):
        if isinstance(kind, VecType):
            space = self.interpreter_state.type_spaces[kind]
            result = []
            for ind in range(len(space.terms)):
                result.append(space.get(ind).get())
            return np.vstack(result)
        return self.embeddings[kind]
            

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
            #Derive the embedding for the type 
            arg_type = kind.arg_type
            ret_type = kind.ret_type
            space = self.interpreter_state.get_type_space(kind)
            table = self.interpreter_state.get_application_table(kind)
            func_pairs = []
            for func_ind in range(len(space.terms)):
                func_row = table.table[func_ind]
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
                func_pair = (arg_embeds, ret_embeds)
                func_pairs.append(func_pair)
            #Now that we have the embedded input, output pairs, 
            #we can get the embedding matrix for this type space
            X = self.get_embed_matrix(arg_type) #Used for convex hull evaluation
            X = np.transpose(X)
            embedding_matrix = get_embedding_matrix(func_pairs, X)
            embedding_matrix = np.transpose(embedding_matrix)
            self.embeddings[kind] = embedding_matrix
