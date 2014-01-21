import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
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

lmbdas = numpy.array([10**-7, 10**-6, 10**-5, 10**-4]) 
aucs = numpy.zeros(lmbdas.shape[0])

r = numpy.ones(X.shape[0])*0.0
eps = 0.000001
sigma = 200
stochastic = True
lmbda = 0.1

maxLocalAuc = MaxLocalAUC(lmbda, k, r, sigma=sigma, eps=eps, stochastic=stochastic)
maxLocalAuc.maxIterations = 500
maxLocalAuc.numRowSamples = 50
maxLocalAuc.numAucSamples = 200
maxLocalAuc.approxDerivative = True
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.recordStep = 50

plotInd = 0
omegaList = maxLocalAuc.getOmegaList(X)

#Now let's vary learning rate sigma 
maxLocalAuc.lmbda = 0.0001

sigmas = numpy.array([10, 50, 100, 200, 400, 800]) 
aucs = numpy.zeros(sigmas.shape[0])

for i, sigma in enumerate(sigmas): 
    maxLocalAuc.sigma = sigma
            
    logging.debug("Starting training")
    U, V = maxLocalAuc.learnModel(X, False)
    logging.debug("Done")
    
    aucs[i] = maxLocalAuc.localAUCApprox(X, U, V, omegaList)
  
logging.debug(aucs)
  
plt.figure(plotInd)
plt.plot(sigmas, aucs)
plt.xlabel("sigma")
plt.ylabel("local AUC")
plotInd += 1

#Now the number of row samples used for stochastic gradient descent
numRowSamplesArr = numpy.array([5, 10, 20, 50, 100])
aucs = numpy.zeros(numRowSamplesArr.shape[0])

maxLocalAuc.lmbda = 0.0001
maxLocalAuc.sigma = 200

for i, numRowSamples in enumerate(numRowSamplesArr): 
    maxLocalAuc.numRowSamples = numRowSamples
            
    logging.debug("Starting training")
    U, V = maxLocalAuc.learnModel(X, False)
    logging.debug("Done")
    
    aucs[i] = maxLocalAuc.localAUCApprox(X, U, V, omegaList)
   
logging.debug(aucs)   
   
plt.figure(plotInd)
plt.plot(numRowSamplesArr, aucs)
plt.xlabel("numRowSamples")
plt.ylabel("local AUC")
plotInd += 1

#Now the number of samples used to approximate AUC
numAucSamplesArr = numpy.array([20, 50, 100, 200])
aucs = numpy.zeros(numAucSamplesArr.shape[0])

maxLocalAuc.lmbda = 0.0001
maxLocalAuc.sigma = 200
maxLocalAuc.numRowSamples = 50

for i, numAucSamples in enumerate(numAucSamplesArr): 
    maxLocalAuc.numAucSamples = numAucSamples
            
    logging.debug("Starting training")
    U, V = maxLocalAuc.learnModel(X, False)
    logging.debug("Done")
    
    aucs[i] = maxLocalAuc.localAUCApprox(X, U, V, omegaList)
   
logging.debug(aucs)   
   
plt.figure(plotInd)
plt.plot(numAucSamplesArr, aucs)
plt.xlabel("numAucSamples")
plt.ylabel("local AUC")
plotInd += 1
plt.show()
