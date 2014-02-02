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
density = 0.05
m = 1000 
n = 1000 
k = 10 
numInds = int(m*n*density)
X = SparseUtils.generateSparseLowRank((m, n), k, numInds)

X.data[X.data >= numpy.mean(X.data)] = 1
X.data[X.data < numpy.mean(X.data)] = 0

X = X.tocsr()


lmbda = 0.000
u = 0.3
eps = 0.001
sigma = 0.2
stochastic = True
maxLocalAuc = MaxLocalAUC(lmbda, k, u, sigma=sigma, eps=eps, stochastic=stochastic)
maxLocalAuc.maxIterations = m*2
maxLocalAuc.numRowSamples = 50
maxLocalAuc.numColSamples = 50
maxLocalAuc.numAucSamples = 50
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 5
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.1    
maxLocalAuc.t0 = 0.1    
omegaList = maxLocalAuc.getOmegaList(X)

logging.debug("Starting training")
ProfileUtils.profile('U, V, objs, aucs, iterations, time = maxLocalAuc.learnModel(X, True)', globals(), locals())
#U, V, objs, aucs, iterations, times = maxLocalAuc.learnModel(X, True)

r = maxLocalAuc.computeR(U, V)
logging.debug("||U||=" + str(numpy.linalg.norm(U)) + " ||V||=" + str(numpy.linalg.norm(V)))
logging.debug("Final local AUC:" + str(maxLocalAuc.localAUCApprox(X, U, V, omegaList, r)))

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