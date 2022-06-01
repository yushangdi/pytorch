import torch
import torch.fx as fx
from functorch import make_fx
from functorch.compile import aot_function, print_compile

import hashlib
import json

# https://stackoverflow.com/questions/3054449/how-to-properly-define-hash-function-for-a-list-of-objects
def fx_hash(arg):
    try:
        return hash(arg)
    except TypeError:
        if(isinstance(arg, torch.fx.immutable_collections.immutable_list) or 
           isinstance(arg, tuple) or
           isinstance(arg, list)):
            hashCode = 1
            for ele in arg:
                hashCode = 31*hashCode + (0 if ele is None else fx_hash(ele)) #TODO: but this works for set, but not for ordered list
            return hashCode
        if(isinstance(arg, torch.fx.immutable_collections.immutable_dict)):
            dhash = hashlib.md5()
            encoded = json.dumps(arg, sort_keys=True).encode()
            dhash.update(encoded)
            return hash(dhash.hexdigest())
        else:
            raise TypeError

def check_same(node, n):
    return node.target == n.target and node.args == n.args and node.kwargs == n.kwargs

def modify(fx_g):
    new_graph = fx.Graph()
    env = {} # map from node in the old graph to node in the new graph
    hash_env = {} # map from the computatio result to a node in the new graph
    for n in fx_g.graph.nodes:
        if n.op == 'placeholder' or n.op == 'output': # != "call_function"
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        elif n.op =='get_attr':
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else: #n.op == 'call_function' or n.op == 'call_module' or n.op == 'call_method'
            # print(n.target)
            # print(n.args)
            # print(n.kwargs)
            # print(n.name) # e.g. cos_1

            try:
            # args = n.args
            # for i in range(len(args)):
            #     print(type(args[i]))
                hash_arg = fx_hash([n.args, n.kwargs]) #TODO: substitute args
                hash_val = (n.target, hash_arg) # (operator, arg) tuple([env[n.args[0]]])
                if hash_val in hash_env and check_same(hash_env[hash_val], n):
                    env[n] = hash_env[hash_val]
                    continue
            except TypeError:
                print("WARNING: type of args is not hashable: {}. Node not checked for CSE".format(n))
                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node #maybe redundant?
                continue
            # new_node = new_graph.call_function(torch.ops.aten.sin, tuple([env[n.args[0]]]))
            new_node = new_graph.node_copy(n, lambda x: env[x])
            hash_env[hash_val] = new_node
            env[n] = new_node
            
    return new_graph





# def f(x):
#     vals = [x]
#     for _ in range(5):
#         vals.append(vals[-1].cos())
#     return vals[-1]

# a = torch.randn(3, 3, requires_grad=True)
# fx_g = make_fx(f)(a)

# aot_function(f, print_compile)(a) 
# # aot_function takes two arguments, a function and something to do after the function.
# # in this case, it prints one forward pass and one backward pass
# # not that in backward pass, torch.ops.aten.t(primals_1) is computed twice
# # it traces the forward and backward graph 

# print(fx_g.graph)
# print(fx_g.code)
# exit(0)

# def f(x):
#     return x.cos().cos()

# fx_g = make_fx(f)(torch.randn(5))

# print(fx_g.graph)
# print(fx_g.code)
# for n in fx_g.graph.nodes:
#     if n.op == 'call_function':
#         if n.target == torch.ops.aten.cos:
#             n.target = torch.ops.aten.sin

# fx_g.recompile()
# print(fx_g.code)
# t = torch.tensor([0])
# print(f(t))
# print(fx_g(t))
# exit(0)

# def f(x):
#     return x.cos().cos()
# fx_g = make_fx(f)(torch.randn(5))

# new_graph = fx.Graph()
# env = {}
# hash_env = {}
# for n in fx_g.graph.nodes:
#     if n.op != 'call_function':
#         new_node = new_graph.node_copy(n, lambda x: env[x])
#         env[n] = new_node
#     elif n.op == 'call_function':
#         hash_val = (n.target, tuple([env[n.args[0]]]))
#         if hash_val in hash_env:
#             env[n] = hash_env[hash_val]
#             continue

#         new_node = new_graph.call_function(torch.ops.aten.sin, tuple([env[n.args[0]]]))
#         hash_env[hash_val] = new_node
#         env[n] = new_node

# print(new_graph)
# t = torch.tensor([0])
# print(f(t))
# print(type(fx_g))
# print(fx_g(t))
# print(fx.GraphModule({"x_1":torch.Tensor},new_graph)(t))