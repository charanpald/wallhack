import array 
import numpy 
import scipy.io
import scipy.sparse.linalg 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt  
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.IdIndexer import IdIndexer
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

for i in range(1, 11): 
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
    u = numpy.sort(u)
    us.append(u)
    
    k = numpy.argmax(numpy.diff(u))
    delta = numpy.max(numpy.diff(u))
    
    ks.append(k)
    
    print("k="+ str(k))
    V = V[:, 0:k]
    
    if i != 1: 
        E = L - Ls[-2]
        E = numpy.array(E.todense())
        EV = E.dot(V)
        L = numpy.array(L.todense())
        
        boundFro.append(numpy.linalg.norm(EV)/delta)
        bound2.append(numpy.linalg.norm(EV, ord=2)/delta)

#2 norm bound is bad 
#Frobenius norm bound is good but only for last few cores 
print(boundFro)
print(bound2)
print(numpy.sqrt(ks))