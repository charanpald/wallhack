import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.Sampling import Sampling
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

"""
Look at convergence with different rate/decay parameters. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
X, U, V = DatasetUtils.syntheticDataset1()
m, n = X.shape

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

u = 5.0/n
w = 1.0 - u
k2 = 8
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k2, w, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 10
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 5
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.5
maxLocalAuc.t0 = 10**-4

alphas = [0.1, 0.2, 0.5, 1.0]
t0s = numpy.logspace(-3, -5, 6, base=10)

for i, t0 in enumerate(t0s): 
    maxLocalAuc.t0 = t0
    logging.debug(maxLocalAuc)
    U, V, trainObjs, trainAucs, testObjs, testAucs, precisions, ind, totalTime = maxLocalAuc.learnModel(trainX, verbose=True)
    
    plt.figure(0)
    plt.plot(trainAucs, label="t0="+str(t0))
    plt.plot(testAucs, label="t0="+str(t0))
    plt.legend()
    
    plt.figure(1)
    plt.plot(trainObjs, label="t0="+str(t0))
    plt.plot(testObjs, label="t0="+str(t0))
    plt.legend()
    
plt.show()

#Using the SVD as the initial solution gives a slightly better solution 