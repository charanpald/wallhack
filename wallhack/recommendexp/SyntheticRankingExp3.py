import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 200
n = 700 
k = 100 
X = SparseUtils.generateSparseBinaryMatrix((m,n), k, csarray=True)
logging.debug("Number of non zero elements: " + str(X.nnz))

trainSplit = 2.0/3

rho = 0.00
u = 0.2
eps = 0.05
sigma = 0.2
stochastic = True
maxLocalAuc = MaxLocalAUC(rho, k, u, sigma=sigma, eps=eps, stochastic=stochastic)

maxLocalAuc.numRowSamples = 50
maxLocalAuc.numColSamples = 50
maxLocalAuc.numAucSamples = 100
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 10
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.5    
maxLocalAuc.t0 = 0.1   
maxLocalAuc.maxIterations = m*10

logging.debug("Splitting into train and test sets")
#trainX, testX = X, X
trainX, testX = SparseUtils.splitNnz(X, trainSplit)
trainOmegaList = maxLocalAuc.getOmegaList(trainX)
testOmegaList = maxLocalAuc.getOmegaList(testX)

logging.debug("Selecting learning rate")
#maxLocalAuc.learningRateSelect(trainX)

logging.debug("Starting model selection")
#maxLocalAuc.modelSelect(trainX)

logging.debug("Starting training")
U, V, objs, aucs, iterations, times = maxLocalAuc.learnModel(trainX, True)

r = maxLocalAuc.computeR(U, V)
logging.debug("||U||=" + str(numpy.linalg.norm(U)) + " ||V||=" + str(numpy.linalg.norm(V)))
logging.debug("Train local AUC:" + str(maxLocalAuc.localAUCApprox(trainX, U, V, trainOmegaList, r)))
logging.debug("Test local AUC:" + str(maxLocalAuc.localAUCApprox(testX, U, V, testOmegaList, r)))

logging.debug("Train Precision@5=" + str(MCEvaluator.precisionAtK(trainX, U, V, 5)))
logging.debug("Train Precision@10=" + str(MCEvaluator.precisionAtK(trainX, U, V, 10)))
logging.debug("Train Precision@20=" + str(MCEvaluator.precisionAtK(trainX, U, V, 20)))
logging.debug("Train Precision@50=" + str(MCEvaluator.precisionAtK(trainX, U, V, 50)))

logging.debug("Test Precision@5=" + str(MCEvaluator.precisionAtK(testX, U, V, 5)))
logging.debug("Test Precision@10=" + str(MCEvaluator.precisionAtK(testX, U, V, 10)))
logging.debug("Test Precision@20=" + str(MCEvaluator.precisionAtK(testX, U, V, 20)))
logging.debug("Test Precision@50=" + str(MCEvaluator.precisionAtK(testX, U, V, 50)))
