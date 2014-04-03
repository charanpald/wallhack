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
n = 1000 
k = 10 
X = SparseUtils.generateSparseBinaryMatrix((m,n), k)
logging.debug("Number of non zero elements: " + str(X.nnz))

trainSplit = 0.8
trainX, testX = SparseUtils.splitNnz(X, trainSplit)

rho = 0.000
u = 0.1
eps = 10**-5
sigma = 0.2
stochastic = True
maxLocalAuc = MaxLocalAUC(k, u, sigma=sigma, eps=eps, stochastic=stochastic)
maxLocalAuc.maxIterations = m*20
maxLocalAuc.numRowSamples = 20
maxLocalAuc.numColSamples = 20
maxLocalAuc.numAucSamples = 50
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.numStepIterations = 500 
maxLocalAuc.recordStep = maxLocalAuc.numStepIterations
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 5.0  
maxLocalAuc.t0 = 0.001 

logging.debug("Starting training")
ProfileUtils.profile('U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, time = maxLocalAuc.learnModel(trainX, testX=X, verbose=True)', globals(), locals())
#U, V, objs, trainAucs, testAucs, iterations, times = maxLocalAuc.learnModel(X, True)

logging.debug("||U||=" + str(numpy.linalg.norm(U)) + " ||V||=" + str(numpy.linalg.norm(V)))
logging.debug("Final local AUC:" + str(MCEvaluator.localAUCApprox(X, U, V, u)))

logging.debug("Number of iterations: " + str(iterations))

plt.figure(0)
plt.plot(trainObjs, label="train")
plt.plot(testObjs, label="test")
plt.xlabel("iteration")
plt.ylabel("objective")

plt.figure(1)
plt.plot(trainAucs, label="train")
plt.plot(testAucs, label="test")
plt.xlabel("iteration")
plt.ylabel("local AUC")
plt.show()