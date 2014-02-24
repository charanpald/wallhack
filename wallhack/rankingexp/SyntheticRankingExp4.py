import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.Sampling import Sampling

"""
Let's figure out when the local AUC on the training set is so different on the full dataset 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(22)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 5 
n = 200 
k = 3 
u = 0.3
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, 1-u, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non-zero elements: " + str(X.nnz))

U = U*s

u = 1.0

trainSplit = 2.0/3
trainX, testX = SparseUtils.splitNnz(X, trainSplit)
cvInds = Sampling.randCrossValidation(3, X.nnz)
#trainInds, testInds = cvInds[0]
#trainX = SparseUtils.submatrix(X, trainInds)
#testX = SparseUtils.submatrix(X, testInds)

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))



logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, u)))
logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, u)))
logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, u)))

u = 0.2
rho = 0.05
k2 = 3
eps = 0.000
sigma = 0.05
stochastic = True
maxLocalAuc = MaxLocalAUC(rho, k2, u, sigma=sigma, eps=eps, stochastic=stochastic)
maxLocalAuc.maxIterations = m*20
maxLocalAuc.numRowSamples = 50
maxLocalAuc.numColSamples = 50
maxLocalAuc.numAucSamples = 50
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 1
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.1    
maxLocalAuc.t0 = 0.1


"""
rhos = numpy.array([ 0.1, 0.05, 0.01, 0.0])
nus = numpy.array([1, 5, 10, 20, 50])

testAucsGrid = numpy.zeros((rhos.shape[0], nus.shape[0]))

numRuns = 3

for a in range(numRuns): 
    #Generate a new training/test set 
    X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, 1-u, csarray=True, verbose=True, indsPerRow=200)
    logging.debug("Number of non-zero elements: " + str(X.nnz))
    U = U*s
    
    #u = 1.0
    trainSplit = 2.0/3
    trainX, testX = SparseUtils.splitNnz(X, trainSplit)
    
    for i, rho in enumerate(rhos): 
        for j, nu in enumerate(nus): 
            maxLocalAuc.rho = rho
            maxLocalAuc.nu = nu
            
            logging.debug("Starting training")
            logging.debug(maxLocalAuc)
            U, V, objs, aucs, testAucs, iterations, times = maxLocalAuc.learnModel(trainX, True, testX=testX)
    
            testAucsGrid[i, j] += MCEvaluator.localAUCApprox(testX, U, V, u, numAucSamples=100)

print(testAucsGrid/numRuns)


print(aucs)
print(testAucs)

"""

print(maxLocalAuc)

U, V, objs, aucs, testAucs, iterations, times = maxLocalAuc.learnModel(trainX, True, testX=testX)

import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 

logging.debug("Train local AUC:" + str(MCEvaluator.localAUCApprox(trainX, U, V, u, numAucSamples=100)))
logging.debug("Test local AUC:" + str(MCEvaluator.localAUCApprox(testX, U, V, u, numAucSamples=100)))


trainMeasures = []
trainMeasures.append(MCEvaluator.precisionAtK(trainX, U, V, 5))
trainMeasures.append(MCEvaluator.precisionAtK(trainX, U, V, 10))
trainMeasures.append(MCEvaluator.precisionAtK(trainX, U, V, 20))
logging.debug(trainMeasures)

testMeasures = []
testMeasures.append(MCEvaluator.precisionAtK(testX, U, V, 5))
testMeasures.append(MCEvaluator.precisionAtK(testX, U, V, 10))
testMeasures.append(MCEvaluator.precisionAtK(testX, U, V, 20))
logging.debug(testMeasures)

plt.plot(aucs)
plt.plot(testAucs)
plt.show()


