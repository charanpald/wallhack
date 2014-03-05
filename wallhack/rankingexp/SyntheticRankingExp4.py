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
n = 200 
k = 16 
w = 0.1
u = 1-w
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, u, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non-zero elements: " + str(X.nnz))

U = U*s

trainSplit = 2.0/3
trainX, testX = SparseUtils.splitNnz(X, trainSplit)
cvInds = Sampling.randCrossValidation(3, X.nnz)

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, w)))
logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, w)))
logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, w)))

#w = 1.0
rho = 0.00
k2 = 16
eps = 0.0001
sigma = 0.05
maxLocalAuc = MaxLocalAUC(rho, k2, w, sigma=sigma, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = m*50
maxLocalAuc.numRowSamples = 100
maxLocalAuc.numStepIterations = 1
maxLocalAuc.numAucSamples = 100
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 50
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 10    
maxLocalAuc.t0 = 0.01


#logging.debug(maxLocalAuc)
#maxLocalAuc.learningRateSelect(X)
#maxLocalAuc.learnModel(X)


w = 0.1
sigma = 50
rho = 0.0001
maxLocalAuc2 = MaxLocalAUC(rho, k2, w, sigma=sigma, eps=eps, stochastic=False)
maxLocalAuc2.maxIterations = m*2
maxLocalAuc2.recordStep = 1
maxLocalAuc.numAucSamples = 100
maxLocalAuc2.rate = "optimal"
maxLocalAuc2.alpha = 5.0    
maxLocalAuc2.t0 = 0.5
maxLocalAuc2.project = True
maxLocalAuc2.nu = 1.0

logging.debug(maxLocalAuc2)
maxLocalAuc2.learnModel(X)
