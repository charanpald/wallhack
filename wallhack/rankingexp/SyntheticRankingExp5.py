import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.Sampling import Sampling
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

"""
Script to see if orthogonality constraint helps. 
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 500
n = 100
k = 10 
u = 0.05
w = 1-u
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non-zero elements: " + str(X.nnz))

U = U*s

trainSplit = 2.0/3
trainX, testX = SparseUtils.splitNnz(X, trainSplit)
cvInds = Sampling.randCrossValidation(3, X.nnz)

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
#logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, w)))
#logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, w)))
#logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, w)))

#w = 1.0
k2 = k
eps = 10**-6
sigma = 10
maxLocalAuc = MaxLocalAUC(k2, w, sigma=sigma, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = m*20
maxLocalAuc.numRowSamples = 20
maxLocalAuc.numStepIterations = 200
maxLocalAuc.numAucSamples = 50
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = maxLocalAuc.numStepIterations
maxLocalAuc.nu = 20
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 5
maxLocalAuc.t0 = 10**-3


rhos = 10.0**numpy.arange(-1, -8, -2)
print(rhos)

for i, rho in enumerate(rhos): 
    maxLocalAuc.rho = rho
    logging.debug(maxLocalAuc)
    U, V, trainObjs, trainAucs, testObjs, testAucs, ind, totalTime = maxLocalAuc.learnModel(X, verbose=True)
    
    plt.figure(0)
    plt.plot(trainAucs, label="rho="+str(rho))
    plt.legend()
    
    plt.figure(1)
    plt.plot(trainObjs, label="rho="+str(rho))
    plt.legend()
    
plt.show()

#Using rho does not seem to help 