import numpy
import logging
import sys
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
n = 500 
k = 10 
numInds = int(m*n*0.1)
X = SparseUtils.generateSparseLowRank((m, n), k, numInds)

X = X/X
X = X.tocsr()


lmbda = 0.000
r = numpy.ones(X.shape[0])*0.0
eps = 0.0001
sigma = 0.1
stochastic = True
maxLocalAuc = MaxLocalAUC(lmbda, k, r, sigma=sigma, eps=eps, stochastic=stochastic)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 50
maxLocalAuc.numAucSamples = 50
maxLocalAuc.iterationsPerUpdate = 50
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 1
maxLocalAuc.rate = "constant"
        
omegaList = maxLocalAuc.getOmegaList(X)

logging.debug("Starting training")
ProfileUtils.profile('U, V, objs, aucs, iterations, time = maxLocalAuc.learnModel(X, True)', globals(), locals())
#U, V, objs, aucs, iterations, times = maxLocalAuc.learnModel(X, True)

logging.debug("||U||=" + str(numpy.linalg.norm(U)) + " ||V||=" + str(numpy.linalg.norm(V)))
logging.debug("Final local AUC:" + str(maxLocalAuc.localAUCApprox(X, U, V, omegaList)))

logging.debug("Number of iterations: " + str(iterations))

plt.figure(0)
plt.plot(objs)
plt.xlabel("iteration")
plt.ylabel("objective")

plt.figure(1)
plt.plot(aucs)
plt.xlabel("iteration")
plt.ylabel("local AUC")
plt.show()