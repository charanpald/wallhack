import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from sandbox.util.ProfileUtils import ProfileUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.Util import Util

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 500
n = 100
k = 10 
u = 0.05
w = 1-u
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non-zero elements: " + str(X.nnz))

U = U*s

trainSplit = 2.0/3
trainX, testX = SparseUtils.splitNnz(X, trainSplit)

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
#logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, w)))
#logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, w)))
#logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, w)))

#w = 1.0
k2 = k
eps = 10**-5
sigma = 10
lmbda = 0.01
maxLocalAuc = MaxLocalAUC(k2, w, sigma=sigma, eps=eps, lmbda=lmbda, stochastic=True)
maxLocalAuc.maxIterations = m*20
maxLocalAuc.numRowSamples = 20
maxLocalAuc.numStepIterations = 200
maxLocalAuc.numAucSamples = 50
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.recordStep = maxLocalAuc.numStepIterations
maxLocalAuc.nu = 20
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.3
maxLocalAuc.t0 = 10**-3
maxLocalAuc.folds = 3
maxLocalAuc.rho = 0.00
maxLocalAuc.ks = 2**numpy.arange(3, 7)

logging.debug("Starting training")
logging.debug(maxLocalAuc)
maxLocalAuc.modelSelect(trainX)
#ProfileUtils.profile('U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, time = maxLocalAuc.learnModel(trainX, testX=X, verbose=True)', globals(), locals())
U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, time = maxLocalAuc.learnModel(trainX, testX=X, verbose=True)

fpr, tpr = MCEvaluator.averageRocCurve(X, U, V)

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

plt.figure(2)
plt.plot(fpr, tpr)
plt.xlabel("mean false positive rate")
plt.ylabel("mean true positive rate")
plt.show()