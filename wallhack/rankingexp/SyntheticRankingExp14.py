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
X = DatasetUtils.syntheticDataset2()
#X = DatasetUtils.movieLens()

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
maxLocalAuc.maxIterations = 10
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numRecordAucSamples = 200
maxLocalAuc.recordStep = 5
maxLocalAuc.parallelStep = 1
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.rate = "constant"
maxLocalAuc.alpha = 0.01
maxLocalAuc.t0 = 0.1
maxLocalAuc.folds = 4
maxLocalAuc.rho = 0.0
maxLocalAuc.lmbdaU = 0.01
maxLocalAuc.lmbdaV = 0.01
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
maxLocalAuc.parallelSGD = False
maxLocalAuc.startAverage = 30
maxLocalAuc.recommendSize = numpy.array([1, 3, 5, 10])

os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
logging.debug(maxLocalAuc)

numRuns = 2 

numpy.random.seed(21)
numRecords = maxLocalAuc.maxIterations/maxLocalAuc.recordStep + 1 
f1s1 = numpy.zeros((numRecords, maxLocalAuc.recommendSize.shape[0], numRuns))
f1s2 = numpy.zeros((numRecords, maxLocalAuc.recommendSize.shape[0], numRuns))

aucs1 = numpy.zeros((numRecords, numRuns)) 
aucs2 = numpy.zeros((numRecords, numRuns)) 

for i in range(numRuns): 
    #First is phi1 version 
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(X, verbose=True)  
    aucs1[:, i] += testMeasures[:, 1]
    f1s1[:, :, i] += testMeasures[:, 2:6]
    
    #Next is phi2 version 
    maxLocalAuc.phi = 1
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(X, verbose=True)  
    aucs2[:, i] += testMeasures[:, 1]
    f1s2[:, :, i] =+ testMeasures[:, 2:6]

f1s1 = f1s1.mean(2)
f1s2 = f1s2.mean(2)
aucs1 = aucs1.mean(1) 
aucs2 = aucs2.mean(1) 


print(f1s1)
print(aucs1)
print(f1s2)
print(aucs2)


plt.figure(0)
plt.plot(aucs1, "k-", label="phi1")
plt.plot(aucs2, "r-", label="phi2")
plt.ylabel("AUC")
plt.xlabel("iteration")
plt.legend()

plt.figure(1)
plt.plot(f1s1[:, 0], "k-", label="F1@" + str(maxLocalAuc.recommendSize[0]))
plt.plot(f1s1[:, 1], "k--", label="F1@" + str(maxLocalAuc.recommendSize[1]))
plt.plot(f1s1[:, 2], "k-.", label="F1@" + str(maxLocalAuc.recommendSize[2]))
plt.plot(f1s1[:, 3], "k:", label="F1@" + str(maxLocalAuc.recommendSize[3]))
plt.plot(f1s2[:, 0], "r-", label="F1@" + str(maxLocalAuc.recommendSize[0]))
plt.plot(f1s2[:, 1], "r--", label="F1@" + str(maxLocalAuc.recommendSize[1]))
plt.plot(f1s2[:, 2], "r-.", label="F1@" + str(maxLocalAuc.recommendSize[2]))
plt.plot(f1s2[:, 3], "r:", label="F1@" + str(maxLocalAuc.recommendSize[3]))

plt.ylabel("F1")
plt.xlabel("iteration")
plt.legend()

plt.show()
