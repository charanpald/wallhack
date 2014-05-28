import numpy
import logging
import sys

import os
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling 

"""
Let's see if we can get the right learning rate on a subsample of rows 
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

os.system('taskset -p 0xffffffff %d' % os.getpid())

#Create a low rank matrix  

m = 500
n = 100
k = 10 
u = 0.1
w = 1-u
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non-zero elements: " + str(X.nnz))

U = U*s

"""
matrixFileName = PathDefaults.getDataDir() + "movielens/ml-100k/u.data" 
data = numpy.loadtxt(matrixFileName)
X = sppy.csarray((numpy.max(data[:, 0]), numpy.max(data[:, 1])), storagetype="row", dtype=numpy.int)
X.put(numpy.array(data[:, 2]>3, numpy.int), numpy.array(data[:, 0]-1, numpy.int32), numpy.array(data[:, 1]-1, numpy.int32), init=True)
X = SparseUtils.pruneMatrix(X, minNnzRows=10, minNnzCols=10)
logging.debug("Read file: " + matrixFileName)
logging.debug("Shape of data: " + str(X.shape))
logging.debug("Number of non zeros " + str(X.nnz))
(m, n) = X.shape

u = 0.1 
w = 1-u
"""

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
#logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, w)))
#logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, w)))
#logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, w)))

#w = 1.0
k2 = 8
eps = 10**-6
alpha = 0.5
maxLocalAuc = MaxLocalAUC(k2, w, alpha=alpha, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 100
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 2000
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.5
maxLocalAuc.t0 = 10**-4
maxLocalAuc.lmbda = 0.1
#maxLocalAuc.numProcesses = 1

maxLocalAuc.t0s = numpy.array([10**-2, 10**-3, 10**-4, 10**-5])
maxLocalAuc.alphas = 2.0**-numpy.arange(-1, 3, 0.5)

newM = 200
modelSelectX = trainX[0:newM, :]

objs1 = maxLocalAuc.learningRateSelect(X)
#objs2 = maxLocalAuc.learningRateSelect(trainX)
#objs3 = maxLocalAuc.learningRateSelect(modelSelectX)


print(objs1)
#print(objs2)
#print(objs3)
