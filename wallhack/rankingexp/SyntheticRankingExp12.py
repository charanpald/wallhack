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
Compare parallel versus non-parallel SGD 
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
numpy.seterr(all="raise")


X, U, V = DatasetUtils.syntheticDataset1()
X2 = DatasetUtils.syntheticDataset2()


m, n = X.shape
u = 0.1 
w = 1-u


#w = 1.0
k = 8
u = 5/float(n)
w = 1-u
eps = 10**-8
lmbda = 10**-3
maxLocalAuc = MaxLocalAUC(k, w, eps=eps, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.maxIterations = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.recordStep = 5
maxLocalAuc.parallelStep = 1
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.rate = "constant"
maxLocalAuc.alpha = 0.1
maxLocalAuc.t0 = 0.1
maxLocalAuc.folds = 4
maxLocalAuc.rho = 0.0
maxLocalAuc.lmbdaU = 0.1
maxLocalAuc.lmbdaV = 0.1
maxLocalAuc.ks = numpy.array([k])
maxLocalAuc.validationSize = 3
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

os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
logging.debug(maxLocalAuc)

#maxLocalAuc.modelSelect(X)

#First is parallel version 
numpy.random.seed(21)
initU, initV = maxLocalAuc.initUV(X)
U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(X, U=initU, V=initV, verbose=True)  

objs1 = trainMeasures[:, 0]
times1 = trainMeasures[:, 2]
print(objs1)
print(times1)

#Now, non parallel version 
maxLocalAuc.parallelSGD = False 
numpy.random.seed(21)
U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(X, U=initU, V=initV, verbose=True)  

objs2 = trainMeasures[:, 0]
times2 = trainMeasures[:, 2]
print(objs2)
print(times2)

#Now look at 2nd dataset 
numpy.random.seed(21)
maxLocalAuc.parallelSGD = True 
initU, initV = maxLocalAuc.initUV(X2)
U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(X2, U=initU, V=initV, verbose=True)  

objs3 = trainMeasures[:, 0]
times3 = trainMeasures[:, 2]
print(objs3)
print(times3)

#Now, non parallel version 
maxLocalAuc.parallelSGD = False 
numpy.random.seed(21)
U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(X2, U=initU, V=initV, verbose=True)  

objs4 = trainMeasures[:, 0]
times4 = trainMeasures[:, 2]
print(objs4)
print(times4)

plt.figure(0)
plt.plot(numpy.arange(0, maxLocalAuc.maxIterations+1, maxLocalAuc.recordStep), objs1, "k-", label="Synthetic1 parallel SGD")
plt.plot(numpy.arange(0, maxLocalAuc.maxIterations+1, maxLocalAuc.recordStep), objs2, "k--", label="Synthetic1 SGD")
plt.plot(numpy.arange(0, maxLocalAuc.maxIterations+1, maxLocalAuc.recordStep), objs3, "k-.", label="Synthetic2 parallel SGD")
plt.plot(numpy.arange(0, maxLocalAuc.maxIterations+1, maxLocalAuc.recordStep), objs4, "k:", label="Synthetic2 SGD")
plt.ylabel("objective")
plt.xlabel("iteration")
plt.legend()


plt.figure(1)
plt.plot(times1, objs1, "k-", label="Synthetic1 parallel SGD")
plt.plot(times2, objs2, "k--", label="Synthetic1 SGD")
plt.plot(times3, objs3, "k-.", label="Synthetic2 parallel SGD")
plt.plot(times4, objs4, "k:", label="Synthetic2 SGD")
plt.ylabel("objective")
plt.xlabel("time (s)")
plt.legend()
plt.show()