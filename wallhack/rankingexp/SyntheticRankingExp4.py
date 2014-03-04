import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.Sampling import Sampling



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(22)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 100 
n = 50 
k = 3 
u = 0.3
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, 1-u, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non-zero elements: " + str(X.nnz))

U = U*s

u = 0.5
trainSplit = 2.0/3
trainX, testX = SparseUtils.splitNnz(X, trainSplit)
cvInds = Sampling.randCrossValidation(3, X.nnz)

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, u)))
logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, u)))
logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, u)))

u = 0.3
rho = 0.00
k2 = 3
eps = 0.001
sigma = 0.05
maxLocalAuc = MaxLocalAUC(rho, k2, u, sigma=sigma, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = m*5
maxLocalAuc.numRowSamples = 50
maxLocalAuc.numStepIterations = 10
maxLocalAuc.numAucSamples = 20
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 50
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 50    
maxLocalAuc.t0 = 0.01


maxLocalAuc.learningRateSelect(X)
maxLocalAuc.learnModel(X)

sigma = 50
maxLocalAuc2 = MaxLocalAUC(rho, k2, u, sigma=sigma, eps=eps, stochastic=False)
maxLocalAuc2.maxIterations = m*2
maxLocalAuc2.recordStep = 10
maxLocalAuc2.rate = "optimal"
maxLocalAuc2.alpha = 0.1    
maxLocalAuc2.t0 = 0.1
maxLocalAuc2.project = True

#maxLocalAuc2.learnModel(X)