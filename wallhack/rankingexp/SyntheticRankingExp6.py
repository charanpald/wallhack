import numpy
import logging
import sys
import sppy 
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


synthetic = False

if synthetic: 
    m = 500
    n = 200
    k = 8 
    u = 20.0/n
    w = 1-u
    X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
    logging.debug("Number of non zero elements: " + str(X.nnz))
    logging.debug("Size of X: " + str(X.shape))
    U = U*s
else: 
    matrixFileName = PathDefaults.getDataDir() + "movielens/ml-100k/u.data" 
    data = numpy.loadtxt(matrixFileName)
    X = sppy.csarray((numpy.max(data[:, 0]), numpy.max(data[:, 1])), storagetype="row", dtype=numpy.int)
    X.put(numpy.array(data[:, 2]>3, numpy.int), numpy.array(data[:, 0]-1, numpy.int32), numpy.array(data[:, 1]-1, numpy.int32), init=True)
    X.prune()
    X = SparseUtils.pruneMatrixRows(X, minNnzRows=10)
    logging.debug("Read file: " + matrixFileName)
    logging.debug("Shape of data: " + str(X.shape))
    logging.debug("Number of non zeros " + str(X.nnz))
    (m, n) = X.shape
    w = 0.9

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
maxLocalAuc.numRowSamples = 10
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.recordStep = 5
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.2
maxLocalAuc.t0 = 10**-2
maxLocalAuc.lmbda = 0.1
#maxLocalAuc.numProcesses = 1
maxLocalAuc.t0s = 10**-numpy.arange(2, 6, 0.5)
maxLocalAuc.alphas = 2.0**-numpy.arange(1, 5, 0.5)

newM = 200
modelSelectX = trainX[0:newM, :]

saveResults = True
outputFile = PathDefaults.getOutputDir() + "ranking/Exp6Results.npz" 

if saveResults: 
    meanObjs1 = maxLocalAuc.learningRateSelect(X)
    meanObjs2 = maxLocalAuc.learningRateSelect(trainX)
    meanObjs3 = maxLocalAuc.learningRateSelect(modelSelectX)

    numpy.savez(outputFile, meanObjs1, meanObjs2, meanObjs3)
else: 
    data = numpy.load(outputFile)
    meanObjs1, meanObjs2, meanObjs3 = data["arr_0"], data["arr_1"], data["arr_2"]
    
    import matplotlib.pyplot as plt 
    plt.contourf(meanObjs1)
    plt.contourf(meanObjs2)
    plt.contourf(meanObjs3)
    plt.show()


print(meanObjs1)
print(meanObjs2)
print(meanObjs3)
