


import numpy
import logging
import sys
import os
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from wallhack.rankingexp.DatasetUtils import DatasetUtils

"""
Test the Rademacher bound on a synthetic dataset. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
numpy.seterr(all="raise")

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
elif dataset == "synthetic2": 
    X = DatasetUtils.syntheticDataset2()
elif dataset == "movielens": 
    X = DatasetUtils.movieLens()
elif dataset == "epinions": 
    X = DatasetUtils.epinions()
    X, userInds = Sampling.sampleUsers2(X, 10000)    
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    X, userInds = Sampling.sampleUsers2(X, 10000)
    
testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

m, n = X.shape


k2 = 8
u2 = 5/float(n)
w2 = 1-u2
eps = 10**-8
lmbda = 0.00
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=0.1, lmbdaV=0.0, stochastic=True)
maxLocalAuc.maxIterations = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.recordStep = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.rate = "constant"
maxLocalAuc.alpha = 0.1
maxLocalAuc.t0 = 1.0
maxLocalAuc.folds = 2
maxLocalAuc.rho = 5.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.validationSize = 3
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.normalise = True
#maxLocalAuc.numProcesses = 1
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.metric = "f1"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.loss = "hinge" 
maxLocalAuc.eta = 0
maxLocalAuc.validationUsers = 1.0
maxLocalAuc.maxNorm = 0.5
maxLocalAuc.bound = True
maxLocalAuc.delta = 0.1


os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
logging.debug(maxLocalAuc)

U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, verbose=True)

print(trainMeasures)

plt.plot(trainMeasures[:, 0], label="train Obj")
plt.plot(testMeasures[:, 0], label="test Obj")
plt.plot(trainMeasures[:, -1], label="bound")
plt.legend()
plt.show()