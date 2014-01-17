import numpy
import logging
import sys
import time
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 50 
n = 20 
k = 5 
numInds = 500
X = SparseUtils.generateSparseLowRank((m, n), k, numInds)

X = X/X
X = X.tocsr()

lmbda = 0.00001
r = numpy.ones(X.shape[0])*0.0
eps = 0.02
sigma = 100
maxLocalAuc = MaxLocalAUC(lmbda, k, r, sigma=sigma, eps=eps, stochastic=False)
        
omegaList = maxLocalAuc.getOmegaList(X)
startTime = time.time()
U, V, objs = maxLocalAuc.learnModel(X, True)
totalTime = time.time() - startTime 

print(numpy.linalg.norm(U), numpy.linalg.norm(V))
logging.debug("Final local AUC:" + str(maxLocalAuc.localAUC(X, U, V, omegaList)))
logging.debug("Total time taken: " + str(totalTime))
logging.debug("Number of iterations: " + str(len(objs)))

print(U)
print(V)
#print(objs)

plt.plot(objs)
plt.show()