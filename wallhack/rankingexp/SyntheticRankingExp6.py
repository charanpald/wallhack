import numpy
import logging
import sys
import sppy
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.Sampling import Sampling
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

"""
Let's increase nu as we go along 
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


k2 = k
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k2, w, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = m*20
maxLocalAuc.numRowSamples = 10
maxLocalAuc.numStepIterations = 500
maxLocalAuc.numAucSamples = 20
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.recordStep = maxLocalAuc.numStepIterations
maxLocalAuc.nu = 1
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.01
maxLocalAuc.t0 = 10**-3
maxLocalAuc.lmbda = 0.001


lmbda = 0.00
nus = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float)**3
print(nus)


numRecordAucSamples = 500 
trainLocalAucs = numpy.zeros(nus.shape[0])
testLocalAucs = numpy.zeros(nus.shape[0])

trainOmegaList = SparseUtils.getOmegaList(trainX)
testOmegaList = SparseUtils.getOmegaList(testX)
U, V = maxLocalAuc.initUV(trainX)
lastInd = 0

for i, nu in enumerate(nus): 
    maxLocalAuc.nu = nu
    maxLocalAuc.nuPrime = nu
    #maxLocalAuc.alpha = maxLocalAuc.alpha/((1 + maxLocalAuc.alpha*maxLocalAuc.t0*lastInd))
    #maxLocalAuc.lmbda = lmbda*nu
    logging.debug(maxLocalAuc)
    U, V, trainObjs, trainAucs, testObjs, testAucs, ind, totalTime = maxLocalAuc.learnModel(trainX, U=U, V=V, verbose=True, testX=testX)
    
    lastInd = ind 
    
    trainLocalAucs[i] = MCEvaluator.localAUCApprox(trainX, U, V, w, numRecordAucSamples, omegaList=trainOmegaList)
    testLocalAucs[i] = MCEvaluator.localAUCApprox(X, U, V, w, numRecordAucSamples, omegaList=testOmegaList)

print(trainLocalAucs)
print(testLocalAucs)