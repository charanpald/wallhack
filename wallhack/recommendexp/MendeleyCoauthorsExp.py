import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from apgl.util.PathDefaults import PathDefaults 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from apgl.util.ProfileUtils import ProfileUtils
from sandbox.util.SparseUtils import SparseUtils
import sppy 
import sppy.io

#import matplotlib
#matplotlib.use("GTK3Agg")
#import matplotlib.pyplot as plt 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

authorAuthorFileName = PathDefaults.getDataDir() + "reference/authorAuthorMatrix.mtx" 
X = sppy.io.mmread(authorAuthorFileName, storagetype="row")
logging.debug("Read file: " + authorAuthorFileName)

X = X[0:500, :]

(m, n) = X.shape
keepInds = numpy.arange(m)[X.power(2).sum(1) != numpy.zeros(m)]
X = X[keepInds, :]

logging.debug("Size of X: " + str(X.shape))
(m, n) = X.shape
k = 10 
trainSplit = 2.0/3

lmbda = 0.000
u = 0.3
eps = 0.001
sigma = 0.2
stochastic = True
maxLocalAuc = MaxLocalAUC(lmbda, k, u, sigma=sigma, eps=eps, stochastic=stochastic)

maxLocalAuc.numRowSamples = 50
maxLocalAuc.numColSamples = 50
maxLocalAuc.numAucSamples = 100
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 5
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.1    
maxLocalAuc.t0 = 0.1   
logging.debug("About to get nonzeros") 


numTrainInds = int(X.nnz*trainSplit)
trainInds = numpy.random.permutation(numTrainInds)[0:numTrainInds]
trainInds = numpy.sort(trainInds)
trainX = SparseUtils.submatrix(X, trainInds)
trainOmegaList = maxLocalAuc.getOmegaList(trainX)

testInds = numpy.setdiff1d(numpy.arange(X.nnz, dtype=numpy.int), trainInds)
testX = SparseUtils.submatrix(X, testInds)
testOmegaList = maxLocalAuc.getOmegaList(testX)

logging.debug("Starting model selection")
sampleSize = 10000
modelSelectInds = numpy.random.permutation(trainX.nnz)[0:sampleSize]
modelSelectInds = numpy.sort(modelSelectInds)
logging.debug("About to get a submatrix")
Xsub = SparseUtils.submatrix(trainX, modelSelectInds)
logging.debug("All done")
maxLocalAuc.maxIterations = m*2
maxLocalAuc.modelSelect(Xsub)

logging.debug("Starting training")
maxLocalAuc.maxIterations = m*2
U, V, objs, aucs, iterations, times = maxLocalAuc.learnModel(trainX, True)

r = maxLocalAuc.computeR(U, V)
logging.debug("||U||=" + str(numpy.linalg.norm(U)) + " ||V||=" + str(numpy.linalg.norm(V)))
logging.debug("Train local AUC:" + str(maxLocalAuc.localAUCApprox(trainX, U, V, trainOmegaList, r)))
logging.debug("Test local AUC:" + str(maxLocalAuc.localAUCApprox(testX, U, V, testOmegaList, r)))
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