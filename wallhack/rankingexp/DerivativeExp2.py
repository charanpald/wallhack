
"""
We want to see how accurate the derivative of V is as we increase the number of samples. 
"""

import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUCCython import derivativeUi, derivativeVi, updateVApprox
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.util.Util import Util
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 500 
n = 1000 
k = 20 
X = SparseUtils.generateSparseBinaryMatrix((m,n), k, 0.95)
logging.debug("Number of non zero elements: " + str(X.nnz))

lmbda = 0.0
numAucSamples = 1000
u = 0.1
sigma = 1
nu = 1 
nuBar = 1
project = False
omegaList = SparseUtils.getOmegaList(X)

U = numpy.random.rand(m, k)
V = numpy.random.rand(n, k)
r = SparseUtilsCython.computeR(U, V, 1-u, numAucSamples)

numPoints = 50
sampleSize = 10 
numAucSamplesList = numpy.linspace(1, 50, numPoints)
norms = numpy.zeros(numPoints)
originalV = V.copy()

for s in range(sampleSize): 
    print(s)
    i = numpy.random.randint(m)
    rowInds = numpy.array([i], numpy.uint)
    vec1 = derivativeVi(X, U, V, omegaList, i, k, lmbda, r)
    vec1 = vec1/numpy.linalg.norm(vec1)   
    
    colInds = numpy.array(numpy.random.permutation(n)[0:100], numpy.uint)
    
    for j, numAucSamples in enumerate(numAucSamplesList): 
        V = originalV.copy()
        updateVApprox(X, U, V, omegaList, rowInds, colInds, numAucSamples, sigma, lmbda, r, nu, nuBar, project)
        
        vec2 = V[i, :] - originalV[i, :]
        
        norms[j] += numpy.abs(numpy.inner(vec1, vec2))

norms /= sampleSize
plt.plot(numAucSamplesList, norms)
plt.show()