import numpy
import logging
import sys
import sppy
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling


"""
Script to see if model selection is the same on a subset of rows or elements 
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

m = 500
n = 200
k = 8 
u = 20.0/n
w = 1-u
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non zero elements: " + str(X.nnz))
logging.debug("Size of X: " + str(X.shape))

"""
matrixFileName = PathDefaults.getDataDir() + "movielens/ml-100k/u.data" 
data = numpy.loadtxt(matrixFileName)
X = sppy.csarray((numpy.max(data[:, 0]), numpy.max(data[:, 1])), storagetype="row", dtype=numpy.int)
X.put(numpy.array(data[:, 2]>3, numpy.int), numpy.array(data[:, 0]-1, numpy.int32), numpy.array(data[:, 1]-1, numpy.int32), init=True)
X = SparseUtils.pruneMatrix(X, minNnzRows=10, minNnzCols=10)
logging.debug("Read file: " + matrixFileName)
logging.debug("Shape of data: " + str(X.shape))
logging.debug("Number of non zeros " + str(X.nnz))

u = 0.1 
w = 1-u
(m, n) = X.shape
"""


testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
#logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, w)))
#logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, w)))
#logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, w)))

#w = 1.0
k2 = 16
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k2, w, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 100
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = m*2
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 1.414
maxLocalAuc.t0 = 0.003
maxLocalAuc.lmbda = 0.01
maxLocalAuc.metric = "precision"


newM = 200
modelSelectX = trainX[0:newM, :]

objs1 = maxLocalAuc.modelSelect(X)
objs2 = maxLocalAuc.modelSelect(trainX)
objs3 = maxLocalAuc.modelSelect(modelSelectX)


print(objs1)
print(objs2)
print(objs3)
