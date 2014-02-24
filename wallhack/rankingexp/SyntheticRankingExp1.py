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


rho = 0.000
u = 0.3
eps = 0.001
sigma = 0.2
stochastic = True
maxLocalAuc = MaxLocalAUC(rho, k, u, sigma=sigma, eps=eps, stochastic=stochastic)
maxLocalAuc.maxIterations = m
maxLocalAuc.numRowSamples = 50
maxLocalAuc.numColSamples = 50
maxLocalAuc.numAucSamples = 50
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 10
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.1    
maxLocalAuc.t0 = 0.1    
omegaList = SparseUtils.getOmegaList(X)

logging.debug("Starting training")
ProfileUtils.profile('U, V, objs, trainAucs, testAucs, iterations, time = maxLocalAuc.learnModel(X, True)', globals(), locals())
#U, V, objs, trainAucs, testAucs, iterations, times = maxLocalAuc.learnModel(X, True)

logging.debug("||U||=" + str(numpy.linalg.norm(U)) + " ||V||=" + str(numpy.linalg.norm(V)))
logging.debug("Final local AUC:" + str(MCEvaluator.localAUCApprox(X, U, V, u)))

logging.debug("Number of iterations: " + str(iterations))

plt.figure(0)
plt.plot(objs)
plt.xlabel("iteration")
plt.ylabel("objective")

plt.figure(1)
plt.plot(trainAucs)
plt.xlabel("iteration")
plt.ylabel("local AUC")
plt.show()