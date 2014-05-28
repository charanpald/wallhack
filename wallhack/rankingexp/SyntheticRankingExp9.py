

import numpy
import logging
import sys
import sppy
import os 
import multiprocessing 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
from sandbox.util.Sampling import Sampling
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

"""
See how regularisation can help overfitting 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

os.system('taskset -p 0xffffffff %d' % os.getpid())

def computeTestAuc(args): 
    trainX, testX, maxLocalAuc  = args 
    
    U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, totalTime = maxLocalAuc.learnModel(trainX, testX=testX, verbose=True)
    trainLocalAuc = trainAucs[-1]
    testLocalAuc = testAucs[-1]
        
    return trainLocalAuc, testLocalAuc 


m = 500
n = 200
k = 8 
u = 20.0/n
w = 1-u
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non zero elements: " + str(X.nnz))
logging.debug("Size of X: " + str(X.shape))

U = U*s

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

eps = 10**-6
maxLocalAuc = MaxLocalAUC(k, w, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 100
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = m
maxLocalAuc.rho = 1.0
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.5
maxLocalAuc.t0 = 10**-3
maxLocalAuc.lmbda = 0.0001


maxItems = 20
lmbdas = 2.0**-numpy.arange(1, 14, 2)

trainLocalAucs = numpy.zeros(lmbdas.shape[0])
testLocalAucs = numpy.zeros(lmbdas.shape[0])

paramList = [] 

for i, lmbda in enumerate(lmbdas):  
    maxLocalAuc.lmbda = lmbda
    logging.debug(maxLocalAuc)
    
    paramList.append((trainX, testX, maxLocalAuc.copy()))

pool = multiprocessing.Pool(processes=7, maxtasksperchild=100)
resultsIterator = pool.imap(computeTestAuc, paramList, 1)
       

for i, lmbda in enumerate(lmbdas): 
    trainLocalAucs[i], testLocalAucs[i] = resultsIterator.next()

pool.terminate()
      
print(lmbdas)      
print(trainLocalAucs)
print(testLocalAucs)

