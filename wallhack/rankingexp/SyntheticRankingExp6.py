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
Test the effect of nu parameter and also quantile threshold u. 
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

m = 500
n = 200
k = 8 
u = 20.0/n
w = 1-u
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non zero elements: " + str(X.nnz))
logging.debug("Size of X: " + str(X.shape))

U = U*s

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
#logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, w)))
#logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, w)))
#logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, w)))

#w = 1.0
k2 = k
eps = 10**-6
sigma = 10
maxLocalAuc = MaxLocalAUC(k2, w, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = m*10
maxLocalAuc.numRowSamples = 10
maxLocalAuc.numStepIterations = 500
maxLocalAuc.numAucSamples = 20
maxLocalAuc.initialAlg = "softimpute"
maxLocalAuc.recordStep = maxLocalAuc.numStepIterations
maxLocalAuc.nu = 20
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.2
maxLocalAuc.t0 = 10**-3


nus = numpy.array([5, 10, 20])
print(nus)

for i, nu in enumerate(nus): 
    maxLocalAuc.nu = nu
    logging.debug(maxLocalAuc)
    U, V, trainObjs, trainAucs, testObjs, testAucs, ind, totalTime = maxLocalAuc.learnModel(trainX, verbose=True, testX=testX)
    
    plt.figure(0)
    plt.plot(trainAucs, label="train nu="+str(nu))
    plt.plot(testAucs, label="test nu="+str(nu))
    plt.legend()
    
    plt.figure(1)
    plt.plot(trainObjs, label="train nu="+str(nu))
    plt.plot(testObjs, label="test nu="+str(nu))
    plt.legend()
    
plt.show()

#Using larger nu helps train AUC, not so much test 
#Using a nu on the kappa makes results slightly worse 