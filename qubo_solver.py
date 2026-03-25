import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch

def qubo(Q):
    '''QUBO solver using Gurobi Optimizer (MAX)'''
    N=len(Q)
    
    # model creation
    m = gp.Model("QUBO")
    
    # silenced print
    m.setParam('OutputFlag', 0)

    # synchronization overhead for GPU
    if torch.cuda.is_available():
        m.setParam('Threads', 1)
    
    # variables creation
    x = m.addMVar(shape=N, vtype=GRB.BINARY, name="x")
    
    # goal: MAXIMIZE
    m.setObjective(x @ Q @ x, GRB.MAXIMIZE)   
    
    # optimization
    m.optimize()
    
    # find the solution
    sol = np.round(x.X).astype(int)
    
    return sol
