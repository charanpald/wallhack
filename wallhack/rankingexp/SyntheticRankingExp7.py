

import numpy
import logging
import sys
import sppy
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
Test the effect of  quantile threshold u. 
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

U = U*s

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

#logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, w)))
#logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, w)))
#logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, w)))

#w = 1.0
u2 = 0.05 
w2 = 1-u2 
k2 = k
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = m*10
maxLocalAuc.numRowSamples = 10
maxLocalAuc.numStepIterations = 500
maxLocalAuc.numAucSamples = 20
maxLocalAuc.initialAlg = "softimpute"
maxLocalAuc.recordStep = maxLocalAuc.numStepIterations
maxLocalAuc.nu = 2
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.2
maxLocalAuc.t0 = 10**-2
maxLocalAuc.lmbda = 0.001

numRecordAucSamples = 200
trainOmegaList = SparseUtils.getOmegaList(trainX)
testOmegaList = SparseUtils.getOmegaList(testX)

maxItems = 10
us = numpy.array([0.5, 0.2, 0.1, 0.05, 0.01])

trainLocalAucs = numpy.zeros(us.shape[0])
trainPrecisions = numpy.zeros(us.shape[0])
trainRecalls = numpy.zeros(us.shape[0])

testLocalAucs = numpy.zeros(us.shape[0])
testPrecisions = numpy.zeros(us.shape[0])
testRecalls = numpy.zeros(us.shape[0])

for i, u in enumerate(us): 
    maxLocalAuc.w = 1-u
    logging.debug(maxLocalAuc)
    U, V, trainObjs, trainAucs, testObjs, testAucs, ind, totalTime = maxLocalAuc.learnModel(trainX, verbose=True, testX=testX)
    
    trainOrderedItems = MCEvaluator.recommendAtk(U, V, maxItems)
    trainLocalAucs[i] = MCEvaluator.localAUCApprox(trainX, U, V, w, numRecordAucSamples, omegaList=trainOmegaList)
    trainPrecisions[i] = MCEvaluator.precisionAtK(trainX, trainOrderedItems, maxItems, omegaList=trainOmegaList)
    trainRecalls[i] = MCEvaluator.recallAtK(trainX, trainOrderedItems, maxItems, omegaList=trainOmegaList)
    
    testLocalAucs[i] = MCEvaluator.localAUCApprox(X, U, V, w, numRecordAucSamples, omegaList=testOmegaList)
    testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, maxItems, trainX)
    testPrecisions[i] = MCEvaluator.precisionAtK(testX, testOrderedItems, maxItems, omegaList=testOmegaList)
    testRecalls[i] = MCEvaluator.recallAtK(testX, testOrderedItems, maxItems, omegaList=testOmegaList)

print(trainLocalAucs)
print(trainPrecisions)
print(trainRecalls)
print("\n")

print(testLocalAucs)
print(testPrecisions)
print(testRecalls)
