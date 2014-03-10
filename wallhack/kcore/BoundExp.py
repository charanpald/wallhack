import array 
import numpy 
import scipy.io
import scipy.sparse.linalg 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt  
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.IdIndexer import IdIndexer
from sandbox.util.Latex import Latex 
from apgl.graph.GraphUtils import GraphUtils 

"""
We try to figure out the change in L_i and L_{i+1}
"""

numpy.set_printoptions(suppress=True, precision=4)

dataDir = PathDefaults.getDataDir() + "kcore/"
indexer = IdIndexer()

node1Inds = array.array("i")
node2Inds = array.array("i")

Ls = []
us = []

boundFro = [] 
bound2 = []
ks = []
eyes = []
deltas = []

for i in range(1, 9): 
    print(i)
    networkFilename = dataDir + "network_1_kcores/network_1-core" + str("%02d" % (i,)) + ".txt"
    
    networkFile = open(networkFilename)
    networkFile.readline()
    networkFile.readline()
    networkFile.readline()
    networkFile.readline()
    
    node1Inds = array.array("i")
    node2Inds = array.array("i")    
    
    for line in networkFile: 
        vals = line.split()
        
        node1Inds.append(indexer.append(vals[0]))
        node2Inds.append(indexer.append(vals[1]))
    
    node1Inds = numpy.array(node1Inds)
    node2Inds = numpy.array(node2Inds)
    
    m = len(indexer.getIdDict())    
    
    A = numpy.zeros((m, m))
    A[node1Inds, node2Inds] = 1
    A = (A+A.T)/2
    
    A = scipy.sparse.csr_matrix(A)
    L = GraphUtils.normalisedLaplacianSym(A)
    Ls.append(L)
    
    u, V = scipy.sparse.linalg.eigs(L, k=m-2, which="SM")
    u = u.real 
    inds = numpy.argsort(u)
    u = u[inds]
    V = V[:, inds]
    us.append(u)
    

    k0 = numpy.where(u > 0.01)[0][0]
    k = numpy.argmax(numpy.diff(u[k0:]))
    
    ks.append(k)
    
    print("k0="+ str(k0) + " k="+ str(k))
    V = V[:, k0:k0+k+1]
    
    if i != 1: 
        E = L - Ls[-2]
        E = numpy.array(E.todense())
        EV = E.dot(V)
        L = numpy.array(L.todense())
        
        delta = us[-2][k0+k+1] - us[-1][k0+k]
        
        boundFro.append(numpy.linalg.norm(EV)/delta)
        bound2.append(numpy.linalg.norm(EV, ord=2)/delta)
        eyes.append(i)
        deltas.append(delta)

boundFro = numpy.array(boundFro)
bound2 = numpy.array(bound2)
eyes = numpy.array(eyes)-1
deltas = numpy.array(deltas)

#2 norm bound is bad 
#Frobenius norm bound is good but only for last few cores
print(1/deltas)
print(ks)
print(boundFro/numpy.sqrt(ks[:-1]))
print(Latex.array1DToRow(eyes)) 
print(Latex.array1DToRow(numpy.sqrt(ks)))
print(Latex.array1DToRow(boundFro))
print(Latex.array1DToRow(bound2)) 
