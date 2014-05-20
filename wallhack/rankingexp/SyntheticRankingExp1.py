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

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)

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
eps = 10**-6
lmbda = 0.03125
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbda=lmbda, stochastic=True)
maxLocalAuc.maxIterations = 100
maxLocalAuc.numRowSamples = 100
maxLocalAuc.numStepIterations = 1000
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numRecordAucSamples = 200
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = maxLocalAuc.numStepIterations*2
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.5
maxLocalAuc.t0 = 0.001
maxLocalAuc.folds = 2
maxLocalAuc.C = 0.001
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.testSize = 5
maxLocalAuc.lmbdas = 2.0**-numpy.arange(0, 10, 2)
#maxLocalAuc.numProcesses = 1
#maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
#maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.alphas = numpy.array([0.5, 0.25])
maxLocalAuc.t0s = numpy.array([0.0001, 0.001])
maxLocalAuc.normalise = True
maxLocalAuc.metric = "precision"

os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
logging.debug(maxLocalAuc)

#modelSelectX = trainX[0:100, :]
#maxLocalAuc.learningRateSelect(trainX)
#maxLocalAuc.modelSelect(trainX)
#ProfileUtils.profile('U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, time = maxLocalAuc.learnModel(trainX, testX=testX, verbose=True)', globals(), locals())

U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, time = maxLocalAuc.learnModel(trainX, testX=testX, verbose=True)


maxItems = 5
p = 5

trainOrderedItems = MCEvaluator.recommendAtk(U, V, maxItems)
testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, maxItems, trainX)

for p in [1, 3, 5]: 
    logging.debug("Train precision@" + str(p) + "=" + str(MCEvaluator.precisionAtK(trainX, trainOrderedItems, p, omegaList=trainOmegaList))) 
    logging.debug("Test precision@" + str(p) + "=" + str(MCEvaluator.precisionAtK(testX, testOrderedItems, p, omegaList=testOmegaList))) 


plt.figure(0)
plt.plot(trainObjs, label="train")
plt.plot(testObjs, label="test") 
plt.xlabel("iteration")
plt.ylabel("objective")
plt.legend()


plt.figure(1)
plt.plot(trainAucs, label="train")
plt.plot(testAucs, label="test")
plt.xlabel("iteration")
plt.ylabel("local AUC")
plt.legend()

#fprTrain, tprTrain = MCEvaluator.averageRocCurve(trainX, U, V)
#fprTest, tprTest = MCEvaluator.averageRocCurve(testX, U, V)
#
#plt.figure(2)
#plt.plot(fprTrain, tprTrain, label="train")
#plt.plot(fprTest, tprTest, label="test")
#plt.xlabel("mean false positive rate")
#plt.ylabel("mean true positive rate")
#plt.legend()
plt.show()

