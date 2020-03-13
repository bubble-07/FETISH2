import numpy as np
from type_ids import VecType, FuncType
from func_impls import *
from basic_terms import *
from embedder import *
from interpreter_support import *
from recommender import *

#Class containing all evaluated terms of a given
#type [indexed, searchable like a map or a set]
class TypeSpace(object):
    def __init__(self, my_type):
        self.my_type = my_type
        self.terms = []
        self.term_to_index_map = {}
    def __eq__(self, other):
        return isinstance(other, TypeSpace) and self.my_type == other.my_type
    def __hash__(self):
        return hash(self.my_type)
    def __str__(self):
        result = "Type Space of Type: " + str(self.my_type) + "\n"
        for i in range(len(self.terms)):
            result += "\t" + str(i) + ": " + str(self.terms[i]) + "\n"
        return result
    #Add a given term to this type space, if it doesn't already
    #exist in here. Returns the index of the term in the space.
    def add(self, term):
        if (not (term in self.term_to_index_map)):
            ind = len(self.terms)
            self.terms.append(term)
            self.term_to_index_map[term] = ind
            return ind
        else:
            return self.term_to_index_map[term]

    #Gets a pointer to the given term
    def get_pointer(self, term):
        return TermPointer(self, self.add(term))

    #Given an index to a term, return the underlying term
    def get(self, ind):
        return self.terms[ind]

#Contains [sparse] memoized data about applications of functions f : T -> S
#to arguments of type t
class ApplicationTable(object):
    def __init__(self, func_space, arg_space, result_space):
        self.func_space = func_space
        self.arg_space = arg_space
        self.result_space = result_space
        self.table = {}
        self.table_inv = {} #The "inverse" of table [from result inds to (func, arg) ind pairs]
    def __eq__(self, other):
        return (isinstance(other, ApplicationTable) and self.func_space == other.func_space)
    def __hash__(self):
        return hash(self.func_space)
    def __str__(self):
        result = "Application Table for: " + str(self.func_space.my_type) + "\n"
        for key in self.table:
            result += "\t" + str(key) + ": " + str(self.table[key]) + "\n"
        return result

    #Returns a generator of index triples (func_ind, arg_ind, ret_ind)
    def get_index_triple_generator(self):
        for func_ind in self.table:
            for arg_ind in self.table[func_ind]:
                ret_ind = self.table[func_ind][arg_ind]
                yield (func_ind, arg_ind, ret_ind)

    def get_num_in_row(self, func_ind):
        if (not (func_ind in self.table)):
            return 0
        return len(self.table[func_ind])

    def link(self, func_ind, arg_ind, result_ind):
        if (not (func_ind in self.table)):
            self.table[func_ind] = {}
        self.table[func_ind][arg_ind] = result_ind

        if (not (result_ind in self.table_inv)):
            self.table_inv[result_ind] = []
        self.table_inv[result_ind].append((func_ind, arg_ind))

    #Returns a list of all known applications in this table which
    #yield the desired return pointer
    def get_term_applications_yielding(self, result_ptr):
        result_ind = result_ptr.get_index()
        if (not (result_ind in table_inv)):
            return []
        raw_result = self.table_inv[result_ind]
        result = []
        for func_ind, arg_ind in raw_result:
            func_ptr = TermPointer(self.func_space, func_ind)
            arg_ptr = TermPointer(self.arg_space, arg_ind)
            application = TermApplication(func_ptr, arg_ptr)
            result.append(application)
        return result

    #Given a TermApplication, determine whether we have data on the term
    def has_computed(self, term_app):
        func_index = term_app.func_ptr.get_index()
        if (not (func_index in self.table)):
            return False
        arg_index = term_app.arg_ptr.get_index()
        return (arg_index in self.table[func_index])
    #Get a term pointer to an already-computed term from a term application
    def get_computed(self, term_app):
        func_index = term_app.func_ptr.get_index()
        arg_index = term_app.arg_ptr.get_index()
        result_index = self.table[func_index][arg_index]
        return TermPointer(self.result_space, result_index)

    #Given a term application and the term which results from evaluating
    #the term application, add the result to the result type-space [if not already present]
    #and link it up properly in this table. Returns a pointer to the result
    def link_computed(self, term_app, result_term):
        result_ind = self.result_space.add(result_term)
        func_ind = term_app.func_ptr.get_index()
        arg_ind = term_app.arg_ptr.get_index()
        self.link(func_ind, arg_ind, result_ind)
        return TermPointer(self.result_space, result_ind)

    def evaluate(self, interpreter_sate, term_app):
        if (self.has_computed(term_app)):
            return self.get_computed(term_app)
        #Otherwise, we gotta do some work. Extract the terms in question
        func_term = term_app.func_ptr.get_term()
        arg_term = term_app.arg_ptr.get_term()
        
        func_impl = None
        func_args = []
        #func_term is either a FuncImpl or a PartiallyAppliedTerm
        if (isinstance(func_term, FuncImpl)):
            func_impl = func_term
        if (isinstance(func_term, PartiallyAppliedTerm)):
            func_impl = func_term.impl
            func_args = list(func_term.partial_args)
        func_args.append(arg_term)

        #Determine if we have enough arguments to evaluate the implementation
        if (func_impl.ready_to_evaluate(func_args)):
            result_term = func_impl.evaluate(interpreter_state, func_args)
        else:
            result_term = PartiallyAppliedTerm(func_impl, func_args)
        return self.link_computed(term_app, result_term)

#Terms are each one of:
#FuncImpl | np.array [1D] | PartiallyAppliedTerm

class InterpreterState(object):
    def __init__(self):
        self.application_tables = {}
        self.type_spaces = {}
        self.ret_type_lookup = {}

    def get_application_table(self, func_type):
        return self.application_tables[func_type]

    #Initialize and return a type space for the given element type
    #if necessary and return it
    def get_type_space(self, elem_type):
        if (not (elem_type in self.type_spaces)):
            result = TypeSpace(elem_type)
            self.type_spaces[elem_type] = result
        else:
            result = self.type_spaces[elem_type]
        return result

    def add_term(self, elem_type, elem_value):
        return self.get_type_space(elem_type).get_pointer(elem_value)

    #Ensures [by evaluating a bunch of stuff] that every
    #application table's rows contain at least two distinct entries
    def ensure_every_application_table_row_filled(self):
        #First make sure that those type-spaces which
        #are vector types are filled with enough vectors
        for kind in self.type_spaces:
            if isinstance(kind, VecType):
                space = self.type_spaces[kind]
                if (len(space.terms) < 2):
                    space.add(VectorTerm(2 * np.ones(kind.n)))
                if (len(space.terms) < 2):
                    space.add(VectorTerm(-2.0 * np.ones(kind.n)))
        #Now, keep trying to derive new terms
        made_progress = True
        while made_progress:
            made_progress = False
            for func_type in self.application_tables:
                table = self.application_tables[func_type]
                arg_type = table.arg_space.my_type
                for i in range(len(table.func_space.terms)):
                    func_term = table.func_space.terms[i]
                    if (table.get_num_in_row(i) < 2 and len(table.arg_space.terms) > 0):
                        js = list(np.random.choice(len(table.arg_space.terms), size=(2,)))
                        for j in js:
                            arg_term = table.arg_space.terms[j]
                            self.blind_apply(func_term, func_type, arg_term, arg_type)
                            made_progress = True

    def __str__(self):
        result = ""
        for kind in self.type_spaces:
            result += str(self.type_spaces[kind]) + "\n"
        for kind in self.application_tables:
            result += str(self.application_tables[kind]) + "\n"
        return result

    #Gets all registered function types with the given return type
    def get_funcs_returning(self, ret_type):
        if (ret_type not in self.ret_type_lookup):
            return set([])
        return self.ret_type_lookup[ret_type]

    #Gets all known terms applications returning the given result pointer
    def get_term_applications_yielding(self, result_ptr):
        func_types = self.get_funcs_returning(result_ptr.get_type())
        results = []
        for func_type in func_types:
            table = self.application_tables[func_type]
            results = results + table.get_term_applications_yielding(result_ptr)
        return results

    def add_ret_type_lookup(self, func_type):
        if (not (func_type.ret_type in self.ret_type_lookup)):
            self.ret_type_lookup[func_type.ret_type] = set([])
        self.ret_type_lookup[func_type.ret_type].add(func_type)

    #Adds and initializes type spaces and application tables
    #for the given element type recursively
    def add_type(self, elem_type):
        elem_space = self.get_type_space(elem_type)
        if (isinstance(elem_type, FuncType)):
            #Need to also initialize an application table for this function type
            arg_space = self.get_type_space(elem_type.arg_type)
            ret_space = self.get_type_space(elem_type.ret_type)
            if (not (elem_type in self.application_tables)):
                self.add_ret_type_lookup(elem_type)
                table = ApplicationTable(elem_space, arg_space, ret_space)
                self.application_tables[elem_type] = table
            #Recursively perform add_type on the arg and ret types
            self.add_type(elem_type.arg_type)
            self.add_type(elem_type.ret_type)

    def apply_ptrs(self, func_ptr, arg_ptr):
        func_type = func_ptr.get_type()
        func_table = self.get_application_table(func_type)
        term_app = TermApplication(func_ptr, arg_ptr)
        result_ptr = func_table.evaluate(self, term_app)
        return result_ptr

    def blind_apply(self, func_val, func_type, arg_val, arg_type):
        func_space = self.get_type_space(func_type)
        arg_space = self.get_type_space(arg_type)
        func_ptr = func_space.get_pointer(func_val)
        arg_ptr = arg_space.get_pointer(arg_val)
        result_ptr = self.apply_ptrs(func_ptr, arg_ptr)
        return result_ptr.get_term()

DIM = 10

scalar_t = VecType(1)
vector_t = VecType(DIM)
unary_vec_func_t = FuncType(vector_t, vector_t)
binary_vec_func_t = FuncType(vector_t, unary_vec_func_t)
unary_scalar_func_t = FuncType(scalar_t, scalar_t)
binary_scalar_func_t = FuncType(scalar_t, unary_scalar_func_t)
map_func_t = FuncType(unary_scalar_func_t, unary_vec_func_t)
vector_to_scalar_func_t = FuncType(vector_t, scalar_t)
reduce_func_t = FuncType(binary_scalar_func_t, FuncType(scalar_t, vector_to_scalar_func_t))
fill_func_t = FuncType(scalar_t, vector_t)

interpreter_state = InterpreterState()

interpreter_state.add_type(map_func_t)
interpreter_state.add_type(fill_func_t)
interpreter_state.add_type(reduce_func_t)
map_ptr = interpreter_state.add_term(map_func_t, MapImpl(DIM))
fill_ptr = interpreter_state.add_term(fill_func_t, FillImpl(DIM))
reduce_ptr = interpreter_state.add_term(reduce_func_t, ReduceImpl(DIM))
head_ptr = interpreter_state.add_term(vector_to_scalar_func_t, HeadImpl(DIM))
rotate_ptr = interpreter_state.add_term(unary_vec_func_t, RotateImpl(DIM))


#Add all element-wise binary operators
for n, kind in [(1, binary_scalar_func_t), (DIM, binary_vec_func_t)]:
    interpreter_state.add_type(kind)
    interpreter_state.add_term(kind, AddImpl(n))
    interpreter_state.add_term(kind, SubImpl(n))
    interpreter_state.add_term(kind, MultImpl(n))
    interpreter_state.add_term(kind, DivImpl(n))

def get_compose_type(n, m, p):
    return FuncType(FuncType(VecType(m), VecType(p)), 
                                FuncType(FuncType(VecType(n), VecType(m)),
                                         FuncType(VecType(n), VecType(p))))


#Add all relevant versions of composition
for n in [1, DIM]:
    for m in [1, DIM]:
        for p in [1, DIM]:
            compose_type = get_compose_type(n, m, p)
            interpreter_state.add_type(compose_type)
            interpreter_state.add_term(compose_type, ComposeImpl(n, m, p))

#Add in constant functions
for n in [1, DIM]:
    for m in [1, DIM]:
        const_func_type = FuncType(VecType(n), FuncType(VecType(m), VecType(n)))
        interpreter_state.add_type(const_func_type)
        interpreter_state.add_term(const_func_type, ConstImpl(n, m))

'''
print str(interpreter_state)
add_two = interpreter_state.blind_apply(AddImpl(1), binary_scalar_func_t,
                                    VectorTerm(np.array([2.0])), scalar_t)
mul_three = interpreter_state.blind_apply(MultImpl(1), binary_scalar_func_t,
                                    VectorTerm(np.array([3.0])), scalar_t)
compose_times_three = interpreter_state.blind_apply(ComposeImpl(1, 1, 1), get_compose_type(1, 1, 1),
                                                     mul_three, unary_scalar_func_t)
plus_two_times_three = interpreter_state.blind_apply(compose_times_three, FuncType(unary_scalar_func_t, unary_scalar_func_t),
                                                     add_two, unary_scalar_func_t)
six = interpreter_state.blind_apply(plus_two_times_three, unary_scalar_func_t,
                                    VectorTerm(np.array([0.0])), scalar_t)

print six

interpreter_state.blind_apply(add_two, unary_scalar_func_t,
                                    VectorTerm(np.array([3.0])), scalar_t)

map_add_two = interpreter_state.blind_apply(MapImpl(DIM), map_func_t,
                                    add_two, unary_scalar_func_t)
evens = VectorTerm(np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]))
shifted_evens = interpreter_state.blind_apply(map_add_two, unary_vec_func_t,
                                    evens, vector_t)

reduce_sum = interpreter_state.blind_apply(ReduceImpl(DIM), reduce_func_t,
                                    AddImpl(1), binary_scalar_func_t)
reduce_sum_init = interpreter_state.blind_apply(reduce_sum, FuncType(scalar_t, vector_to_scalar_func_t),
                                                VectorTerm(np.array([0.0])), scalar_t)
the_sum = interpreter_state.blind_apply(reduce_sum_init, vector_to_scalar_func_t,
                                        shifted_evens, vector_t)

print str(interpreter_state)

'''
interpreter_state.ensure_every_application_table_row_filled()
print str(interpreter_state)
embedder_state = EmbedderState(interpreter_state)
embedder_state.refresh()
print str(embedder_state)
recommender_state = RecommenderState(embedder_state)

target_kind = FuncType(VecType(1), VecType(1))
scalar_space = interpreter_state.get_type_space(VecType(1))

zero_ind = scalar_space.add(VectorTerm(np.array([0])))
two_ind = scalar_space.add(VectorTerm(np.array([2])))
five_ind = scalar_space.add(VectorTerm(np.array([5])))
four_ind = scalar_space.add(VectorTerm(np.array([4])))
ten_ind = scalar_space.add(VectorTerm(np.array([10])))

target_row = {}
target_row[zero_ind] = zero_ind
target_row[two_ind] = five_ind
target_row[four_ind] = ten_ind

embedder_state.add_target(target_kind, target_row)

for i in range(100):
    recommender_state.step(target_kind, 0)

print str(interpreter_state)
