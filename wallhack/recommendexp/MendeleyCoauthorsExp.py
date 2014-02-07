import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from apgl.util.PathDefaults import PathDefaults 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from apgl.util.ProfileUtils import ProfileUtils
import sppy 
import sppy.io

#import matplotlib
#matplotlib.use("GTK3Agg")
#import matplotlib.pyplot as plt 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

authorAuthorFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix.mtx" 
X = sppy.io.mmread(authorAuthorFileName)
logging.debug("Read file: " + authorAuthorFileName)

X = X[0:500, :]

(m, n) = X.shape
keepInds = numpy.arange(m)[X.power(2).sum(1) != numpy.zeros(m)]
X = X[keepInds, :]

logging.debug("Size of X: " + str(X.shape))
(m, n) = X.shape
k = 10 

lmbda = 0.000
u = 0.3
eps = 0.001
sigma = 0.2
stochastic = True
maxLocalAuc = MaxLocalAUC(lmbda, k, u, sigma=sigma, eps=eps, stochastic=stochastic)
maxLocalAuc.maxIterations = m*2
maxLocalAuc.numRowSamples = 50
maxLocalAuc.numColSamples = 50
maxLocalAuc.numAucSamples = 100
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 5
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.1    
maxLocalAuc.t0 = 0.1   
logging.debug("About to get nonzeros") 
omegaList = maxLocalAuc.getOmegaList(X)

logging.debug("Starting training")
#ProfileUtils.profile('U, V, objs, aucs, iterations, time = maxLocalAuc.learnModel(X, True)', globals(), locals())
U, V, objs, aucs, iterations, times = maxLocalAuc.learnModel(X, True)

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