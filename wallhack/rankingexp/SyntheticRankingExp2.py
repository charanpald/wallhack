import numpy
import logging
import sys
import os
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from sandbox.util.ProfileUtils import ProfileUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.Util import Util
from sandbox.util.Sampling import Sampling
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython

"""
How much random sampling do we need for fast convergence 
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
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

trainOmegaList = SparseUtils.getOmegaList(trainX)
testOmegaList = SparseUtils.getOmegaList(testX)
numRecordAucSamples = 200
logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
logging.debug("Train local AUC:" + str(MCEvaluator.localAUCApprox(trainX, U, V, w, numRecordAucSamples)))
logging.debug("Test local AUC:" + str(MCEvaluator.localAUCApprox(X, U, V, w, numRecordAucSamples, omegaList=testOmegaList)))

#w = 1.0
k2 = k
u2 = 5.0/n
w2 = 1-u2
eps = 10**-15
lmbda = 0
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbda=lmbda, stochastic=True)
maxLocalAuc.maxIterations = m*50
maxLocalAuc.numRowSamples = 100
maxLocalAuc.numStepIterations = 1000
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numRecordAucSamples = 200
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = maxLocalAuc.numStepIterations
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.5
maxLocalAuc.t0 = 0.0001
maxLocalAuc.folds = 2
maxLocalAuc.rho = 0.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.testSize = 3
maxLocalAuc.lmbdas = 2.0**-numpy.arange(0, 10, 2)
#maxLocalAuc.numProcesses = 1
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 6, 1)
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.normalise = True

os.system('taskset -p 0xffffffff %d' % os.getpid())

numRowSamplesArray = numpy.array([10, 20, 50, 100])
numAucSamplesArray = numpy.array([10, 20, 50, 100])

maxLocalAuc.numRowSamples = 100
maxLocalAuc.numAucSamples = 100

for numRowSamples in numRowSamplesArray: 
    maxLocalAuc.numRowSamples = numRowSamples
    logging.debug(maxLocalAuc)
    U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, time = maxLocalAuc.learnModel(trainX, testX=testX, verbose=True)
    
    plt.figure(0)
    plt.plot(trainObjs, label="train nRow="+str(numRowSamples))
    #plt.plot(testObjs, label="test nRow="+str(numRowSamples))
    
    plt.figure(1)
    plt.plot(trainAucs, label="train nRow="+str(numRowSamples))
    #plt.plot(testAucs, label="test nRow="+str(numRowSamples))

maxLocalAuc.numRowSamples = 100
maxLocalAuc.numAucSamples = 100

for numAucSamples in numAucSamplesArray: 
    maxLocalAuc.numAucSamples = numAucSamples
    logging.debug(maxLocalAuc)
    U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, time = maxLocalAuc.learnModel(trainX, testX=testX, verbose=True)

    plt.figure(2)
    plt.plot(trainObjs, label="train nAuc="+str(numAucSamples))
    #plt.plot(testObjs, label="test nAuc="+str(numAucSamples))
    
    plt.figure(3)
    plt.plot(trainAucs, label="train nAuc="+str(numAucSamples))
    #plt.plot(testAucs, label="test nAuc="+str(numAucSamples))


plt.figure(0)
plt.xlabel("iteration")
plt.ylabel("objective")
plt.legend()


plt.figure(1)
plt.xlabel("iteration")
plt.ylabel("local AUC")
plt.legend()

plt.figure(2)
plt.xlabel("iteration")
plt.ylabel("objective")
plt.legend()


plt.figure(3)
plt.xlabel("iteration")
plt.ylabel("local AUC")
plt.legend()

plt.show()

