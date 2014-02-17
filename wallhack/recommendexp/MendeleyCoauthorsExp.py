import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from apgl.util.PathDefaults import PathDefaults 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator
import sppy 
import sppy.io

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

authorAuthorFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix.mtx" 
X = sppy.io.mmread(authorAuthorFileName, storagetype="row")
logging.debug("Read file: " + authorAuthorFileName)

X = X[0:1000, :]

(m, n) = X.shape
logging.debug("Size of X: " + str(X.shape))
logging.debug("Number of non zeros: " + str(X.nnz))

k = 50 
trainSplit = 2.0/3

rho = 0.0001
u = 0.3
eps = 0.01
sigma = 0.2
stochastic = True
maxLocalAuc = MaxLocalAUC(rho, k, u, sigma=sigma, eps=eps, stochastic=stochastic)

maxLocalAuc.numRowSamples = 50
maxLocalAuc.numColSamples = 50
maxLocalAuc.numAucSamples = 100
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 10
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.1    
maxLocalAuc.t0 = 0.1   
maxLocalAuc.maxIterations = m*10

logging.debug("Splitting into train and test sets")
trainX, testX = SparseUtils.splitNnz(X, trainSplit)
trainOmegaList = maxLocalAuc.getOmegaList(trainX)
testOmegaList = maxLocalAuc.getOmegaList(testX)

logging.debug("Selecting learning rate")
#maxLocalAuc.learningRateSelect(trainX)

logging.debug("Starting model selection")
maxLocalAuc.modelSelect(trainX)

logging.debug("Starting training")
U, V, objs, aucs, iterations, times = maxLocalAuc.learnModel(trainX, True)

r = maxLocalAuc.computeR(U, V)
logging.debug("||U||=" + str(numpy.linalg.norm(U)) + " ||V||=" + str(numpy.linalg.norm(V)))
logging.debug("Train local AUC:" + str(maxLocalAuc.localAUCApprox(trainX, U, V, trainOmegaList, r)))

logging.debug("Train Precision@5=" + str(MCEvaluator.precisionAtK(trainX, U, V, 5)))
logging.debug("Train Precision@10=" + str(MCEvaluator.precisionAtK(trainX, U, V, 10)))
logging.debug("Train Precision@20=" + str(MCEvaluator.precisionAtK(trainX, U, V, 20)))
logging.debug("Train Precision@50=" + str(MCEvaluator.precisionAtK(trainX, U, V, 50)))

logging.debug("Test Precision@5=" + str(MCEvaluator.precisionAtK(testX, U, V, 5)))
logging.debug("Test Precision@10=" + str(MCEvaluator.precisionAtK(testX, U, V, 10)))
logging.debug("Test Precision@20=" + str(MCEvaluator.precisionAtK(testX, U, V, 20)))
logging.debug("Test Precision@50=" + str(MCEvaluator.precisionAtK(testX, U, V, 50)))

