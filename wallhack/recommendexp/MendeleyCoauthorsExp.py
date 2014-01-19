import numpy
import logging
import sys
import time
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from apgl.util.ProfileUtils import ProfileUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 500 
n = 100 
k = 5 
numInds = 10000
X = SparseUtils.generateSparseLowRank((m, n), k, numInds)

X = X/X
X = X.tocsr()

lmbda = 0.00001
r = numpy.ones(X.shape[0])*0.0
eps = 0.1
sigma = 200
stochastic = True
maxLocalAuc = MaxLocalAUC(lmbda, k, r, sigma=sigma, eps=eps, stochastic=stochastic)
maxLocalAuc.maxIterations = 200
maxLocalAuc.numSamples = 100
maxLocalAuc.approxDerivative = True
maxLocalAuc.initialAlg = "svd"
        
omegaList = maxLocalAuc.getOmegaList(X)
startTime = time.time()

logging.debug("Starting training")
ProfileUtils.profile('U, V, objs, aucs, iterations = maxLocalAuc.learnModel(X, True)', globals(), locals())
#U, V, objs, aucs, iterations = maxLocalAuc.learnModel(X, True)
totalTime = time.time() - startTime 

logging.debug("||U||=" + str(numpy.linalg.norm(U)) + " ||V||=" + str(numpy.linalg.norm(V)))
logging.debug("Final local AUC:" + str(maxLocalAuc.localAUC(X, U, V, omegaList)))
logging.debug("Total time taken: " + str(totalTime))
logging.debug("Number of iterations: " + str(iterations))

#print(U)
#print(V)
#print(objs)

plt.figure(0)
plt.plot(objs)
plt.xlabel("iteration")
plt.ylabel("objective")

plt.figure(1)
plt.plot(aucs)
plt.xlabel("iteration")
plt.ylabel("local AUC")
plt.show()