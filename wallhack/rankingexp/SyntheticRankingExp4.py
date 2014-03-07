import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.Sampling import Sampling
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 100
n = 200 
k = 16 
w = 0.1
u = 1-w
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, u, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non-zero elements: " + str(X.nnz))

U = U*s

trainSplit = 2.0/3
trainX, testX = SparseUtils.splitNnz(X, trainSplit)
cvInds = Sampling.randCrossValidation(3, X.nnz)

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, w)))
logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, w)))
logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, w)))

#w = 1.0
rho = 0.0
k2 = 16
eps = 0.000001
sigma = 10
maxLocalAuc = MaxLocalAUC(rho, k2, w, sigma=sigma, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = m*50
maxLocalAuc.numRowSamples = 100
maxLocalAuc.numStepIterations = 50
maxLocalAuc.numAucSamples = 50
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 50
maxLocalAuc.nu = 1
maxLocalAuc.rate = "constant"
maxLocalAuc.alpha = m*5
maxLocalAuc.t0 = 0.001


logging.debug(maxLocalAuc)
#maxLocalAuc.learningRateSelect(X)
U, V, objs, trainAucs, testAucs, ind, totalTime = maxLocalAuc.learnModel(X, verbose=True)

plt.figure(0)
plt.plot(trainAucs)
plt.figure(1)
plt.plot(objs)
plt.show()

w = 0.1
sigma = 50
rho = 0.0001
maxLocalAuc2 = MaxLocalAUC(rho, k2, w, sigma=sigma, eps=eps, stochastic=False)
maxLocalAuc2.maxIterations = m*2
maxLocalAuc2.recordStep = 1
maxLocalAuc2.numAucSamples = 100

maxLocalAuc2.initialAlg = "svd"
maxLocalAuc2.rate = "optimal"
maxLocalAuc2.alpha = 1.0    
maxLocalAuc2.t0 = 0.5
maxLocalAuc2.project = True
maxLocalAuc2.nu = 5.0

V = numpy.random.rand(X.shape[1], k)
#logging.debug(maxLocalAuc2)
#maxLocalAuc2.learnModel(X, U=U*s, V=V)
#maxLocalAuc2.learnModel(X) 