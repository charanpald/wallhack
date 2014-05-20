import numpy
import logging
import sys
import sppy
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

"""
Let's see if we can get the right learning rate on a subsample of rows 
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
"""
m = 500
n = 100
k = 10 
u = 0.05
w = 1-u
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non-zero elements: " + str(X.nnz))

U = U*s
"""

matrixFileName = PathDefaults.getDataDir() + "movielens/ml-100k/u.data" 
data = numpy.loadtxt(matrixFileName)
X = sppy.csarray((numpy.max(data[:, 0]), numpy.max(data[:, 1])), storagetype="row")
X[data[:, 0]-1, data[:, 1]-1] = numpy.array(data[:, 2]>3, numpy.int)
logging.debug("Read file: " + matrixFileName)
logging.debug("Shape of data: " + str(X.shape))
logging.debug("Number of non zeros " + str(X.nnz))

u = 0.1 
w = 1-u
(m, n) = X.shape


testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
#logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, w)))
#logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, w)))
#logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, w)))

#w = 1.0
k2 = 16
eps = 10**-6
alpha = 10
maxLocalAuc = MaxLocalAUC(k2, w, alpha=alpha, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 100
maxLocalAuc.numStepIterations = 1000
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = maxLocalAuc.numStepIterations*2
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.5
maxLocalAuc.t0 = 10**-4
maxLocalAuc.folds = 2

maxLocalAuc.ks = numpy.array([4, 8, 16])
maxLocalAuc.lmbdas = numpy.array([0.01])

newM = 200
modelSelectX = trainX[0:newM, :]

meanTestLocalAucs1, stdTestLocalAucs = maxLocalAuc.modelSelect(trainX)
meanTestLocalAucs2, stdTestLocalAucs = maxLocalAuc.modelSelect(modelSelectX)

#Now vary lmbdas
maxLocalAuc.ks = numpy.array([8])
maxLocalAuc.lmbdas = numpy.array([10**-2, 10**-3, 10**-4])

meanTestLocalAucs3, stdTestLocalAucs = maxLocalAuc.modelSelect(trainX)
meanTestLocalAucs4, stdTestLocalAucs = maxLocalAuc.modelSelect(modelSelectX)

print(meanTestLocalAucs1)
print(meanTestLocalAucs2)
print(meanTestLocalAucs3)
print(meanTestLocalAucs4)

