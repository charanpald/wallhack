import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
In this script we play around with parameters to see their effect on convergence speed. 
"""

#Create a low rank matrix  
m = 1000 
n = 2000 
k = 10 
numInds = 50000
X = SparseUtils.generateSparseLowRank((m, n), k, numInds)

X = X/X
X = X.tocsr()



u = 0.2
eps = 0.000001
sigma = 200
stochastic = True
rho = 0.1

maxLocalAuc = MaxLocalAUC(rho, k, u, sigma=sigma, eps=eps, stochastic=stochastic)
maxLocalAuc.maxIterations = 100
maxLocalAuc.numRowSamples = 50
maxLocalAuc.numAucSamples = 200
maxLocalAuc.approxDerivative = True
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.recordStep = 50

plotInd = 0
omegaList = SparseUtils.getOmegaList(X)

#Now let's vary learning rate sigma 
maxLocalAuc.lmbda = 0.0001

sigmas = numpy.array([50, 100, 200, 400, 800, 1600]) 
aucs = numpy.zeros(sigmas.shape[0])
times = numpy.zeros(sigmas.shape[0])

for i, sigma in enumerate(sigmas): 
    maxLocalAuc.sigma = sigma
            
    logging.debug("Starting training")
    U, V, objs, localAucs, iterations, time = maxLocalAuc.learnModel(X, True)
    logging.debug("Done")
    
    aucs[i] = MCEvaluator.localAUCApprox(X, U, V, u)
    times[i] = time
  
logging.debug(aucs)
logging.debug(times)
  
plt.figure(plotInd)
plt.plot(sigmas, aucs)
plt.xlabel("sigma")
plt.ylabel("local AUC")
plotInd += 1

plt.figure(plotInd)
plt.plot(sigmas, times)
plt.xlabel("sigma")
plt.ylabel("time")
plotInd += 1

#Now the number of row samples used for stochastic gradient descent
numRowSamplesArr = numpy.array([5, 10, 20, 50, 100])
aucs = numpy.zeros(numRowSamplesArr.shape[0])
times = numpy.zeros(numRowSamplesArr.shape[0])

maxLocalAuc.lmbda = 0.0001
maxLocalAuc.sigma = 800
maxLocalAuc.numRowSamples = 50
maxLocalAuc.numAucSamples = 200

for i, numRowSamples in enumerate(numRowSamplesArr): 
    maxLocalAuc.numRowSamples = numRowSamples
            
    logging.debug("Starting training")
    U, V, objs, localAucs, iterations, time = maxLocalAuc.learnModel(X, True)
    logging.debug("Done")
    
    aucs[i] = MCEvaluator.localAUCApprox(X, U, V, u)
    times[i] = time
   
logging.debug(aucs)   
logging.debug(times)
   
plt.figure(plotInd)
plt.plot(numRowSamplesArr, aucs)
plt.xlabel("numRowSamples")
plt.ylabel("local AUC")
plotInd += 1

plt.figure(plotInd)
plt.plot(numRowSamplesArr, times)
plt.xlabel("numRowSamples")
plt.ylabel("time")
plotInd += 1

#Now the number of samples used to approximate AUC
numAucSamplesArr = numpy.array([20, 50, 100, 200])
aucs = numpy.zeros(numAucSamplesArr.shape[0])
times = numpy.zeros(numAucSamplesArr.shape[0])

maxLocalAuc.lmbda = 0.0001
maxLocalAuc.sigma = 800
maxLocalAuc.numRowSamples = 50
maxLocalAuc.numAucSamples = 200

for i, numAucSamples in enumerate(numAucSamplesArr): 
    maxLocalAuc.numAucSamples = numAucSamples
            
    logging.debug("Starting training")
    U, V, objs, localAucs, iterations, time = maxLocalAuc.learnModel(X, True)
    logging.debug("Done")
    
    aucs[i] = MCEvaluator.localAUCApprox(X, U, V, u)
    times[i] = time
   
logging.debug(aucs)   
logging.debug(times)
   
plt.figure(plotInd)
plt.plot(numAucSamplesArr, aucs)
plt.xlabel("numAucSamples")
plt.ylabel("local AUC")
plotInd += 1

plt.figure(plotInd)
plt.plot(numAucSamplesArr, times)
plt.xlabel("numAucSamples")
plt.ylabel("time")
plotInd += 1
plt.show()
