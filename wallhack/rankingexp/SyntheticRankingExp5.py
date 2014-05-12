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
Script to see if varying rho helps
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
maxLocalAuc.maxIterations = m*20
maxLocalAuc.numRowSamples = 100
maxLocalAuc.numStepIterations = 1000
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.recordStep = maxLocalAuc.numStepIterations
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.2
maxLocalAuc.t0 = 10**-3


rhos = 10.0**numpy.arange(-1, -8, -2)
print(rhos)

for i, rho in enumerate(rhos): 
    maxLocalAuc.rho = rho
    logging.debug(maxLocalAuc)
    U, V, trainObjs, trainAucs, testObjs, testAucs, ind, totalTime = maxLocalAuc.learnModel(trainX, verbose=True, testX=testX)
    
    plt.figure(0)
    plt.plot(trainAucs, label="train rho="+str(rho))
    plt.plot(testAucs, label="test rho="+str(rho))
    plt.legend()
    
    plt.figure(1)
    plt.plot(trainObjs, label="train rho="+str(rho))
    plt.plot(testObjs, label="test rho="+str(rho))
    plt.legend()
    
plt.show()

#Using rho does not seem to help 