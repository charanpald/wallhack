import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator


"""
Let's figure out when the local AUC on the training set is so different on the full dataset 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 4 
n = 200 
k = 10 
u = 0.3
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, 1-u, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non-zero elements: " + str(X.nnz))

U = U*s

trainSplit = 2.0/3
trainX, testX = SparseUtils.splitNnz(X, trainSplit)
logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, u)))
logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, u)))
logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, u)))

rho = 0.1
k2 = 10
eps = 0.000
sigma = 0.2
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
omegaList = SparseUtils.getOmegaList(X)

logging.debug("Starting training")
logging.debug(maxLocalAuc)
U, V, objs, aucs, iterations, times = maxLocalAuc.learnModel(trainX, True)

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