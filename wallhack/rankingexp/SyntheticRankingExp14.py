import numpy
import logging
import sys
import os
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from wallhack.rankingexp.DatasetUtils import DatasetUtils

"""
Let's look at F1 scores as we go along for different top-n lists
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
numpy.seterr(all="raise")


#X, U, V = DatasetUtils.syntheticDataset1()
#X = DatasetUtils.syntheticDataset2()
X = DatasetUtils.movieLens()

m, n = X.shape
u = 0.1 
w = 1-u


#w = 1.0
k = 8
u = 1.0
w = 1-u
eps = 10**-8
lmbda = 10**-3
maxLocalAuc = MaxLocalAUC(k, w, eps=eps, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.maxIterations = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numRecordAucSamples = 200
maxLocalAuc.recordStep = 5
maxLocalAuc.parallelStep = 1
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.rate = "constant"
maxLocalAuc.alpha = 0.1
maxLocalAuc.t0 = 0.1
maxLocalAuc.folds = 4
maxLocalAuc.rho = 0.0
maxLocalAuc.lmbdaU = 0.1
maxLocalAuc.lmbdaV = 0.1
maxLocalAuc.ks = numpy.array([k])
maxLocalAuc.validationSize = 3
maxLocalAuc.validationUsers = 1.0
maxLocalAuc.z = 10
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.normalise = True
maxLocalAuc.numProcesses = 8
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.metric = "f1"
maxLocalAuc.sampling = "uniform"
maxLocalAuc.itemExpP = 0.5
maxLocalAuc.itemExpQ = 0.5
maxLocalAuc.itemFactors = False
maxLocalAuc.parallelSGD = True
maxLocalAuc.startAverage = 30
maxLocalAuc.recommendSize = numpy.array([1, 3, 5, 10])

os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
logging.debug(maxLocalAuc)

#maxLocalAuc.modelSelect(X)

#First is parallel version 
numpy.random.seed(21)
initU, initV = maxLocalAuc.initUV(X)
U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(X, U=initU, V=initV, verbose=True)  


objs1 = trainMeasures[:, 0]
aucs = testMeasures[:, 1]
f1s = testMeasures[:, 2:6]
print(objs1)
print(f1s)
print(aucs)

plt.figure(0)
plt.plot(objs1, "k-")
plt.ylabel("obj")
plt.xlabel("iteration")
plt.legend()

plt.figure(1)
plt.plot(testMeasures[:, 2], "k-", label="F1@" + str(maxLocalAuc.recommendSize[0]))
plt.plot(testMeasures[:, 3], "k--", label="F1@" + str(maxLocalAuc.recommendSize[1]))
plt.plot(testMeasures[:, 4], "k-.", label="F1@" + str(maxLocalAuc.recommendSize[2]))
plt.plot(testMeasures[:, 5], "k:", label="F1@" + str(maxLocalAuc.recommendSize[3]))
plt.plot(aucs, "r-", label="AUC")
plt.ylabel("F1")
plt.xlabel("iteration")
plt.legend()

plt.show()
