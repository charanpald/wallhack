import numpy
import logging
import sys
import os
import multiprocessing 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
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
    dataset = "movielens"

saveResults = True
prefix = "Rademacher"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 
X = DatasetUtils.getDataset(dataset, nnz=20000)
    

m, n = X.shape

k2 = 16
u2 = 5/float(n)
w2 = 1-u2
eps = 10**-8
lmbda = 0.0
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=lmbda, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.alpha = 0.1
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.bound = True
maxLocalAuc.delta = 0.1
maxLocalAuc.eta = 0
maxLocalAuc.folds = 2
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([4, 8, 16, 32, 64, 128])
maxLocalAuc.lmbdas = 2.0**-numpy.arange(1, 10)
maxLocalAuc.loss = "hinge" 
maxLocalAuc.maxIterations = 100
maxLocalAuc.maxNorm = 1/numpy.sqrt(2)
maxLocalAuc.metric = "f1"
maxLocalAuc.normalise = True
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numProcesses = 1
maxLocalAuc.numRecordAucSamples = 200
maxLocalAuc.numRowSamples = 30
maxLocalAuc.rate = "constant"
maxLocalAuc.recordStep = 10
maxLocalAuc.reg = False
maxLocalAuc.rho = 1.0
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationSize = 5
maxLocalAuc.validationUsers = 1.0

os.system('taskset -p 0xffffffff %d' % os.getpid())

def computeBound(args): 
    X, maxLocalAuc, U, V = args 
    #numpy.random.seed(21)
    logging.debug(maxLocalAuc)
    maxLocalAuc.learningRateSelect(X)
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(X, U=U, V=V, verbose=True)
            
    return trainMeasures[-1, 0], trainMeasures[-1, -1], testMeasures[-1, 0]

folds = 3

if saveResults: 
    paramList = []
    chunkSize = 1
    
    trainObjs = numpy.zeros((maxLocalAuc.lmbdas.shape[0], folds))
    testObjs = numpy.zeros((maxLocalAuc.lmbdas.shape[0], folds))
    bounds = numpy.zeros((maxLocalAuc.lmbdas.shape[0], folds))    
    
    for i in range(folds):
        for lmbda in maxLocalAuc.lmbdas: 
            learner = maxLocalAuc.copy()
            learner.lmbdaU = lmbda
            learner.lmbdaV = lmbda
            U, V = learner.initUV(X)
            paramList.append((X, learner, U.copy(), V.copy()))

    pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
    resultsIterator = pool.imap(computeBound, paramList, chunkSize)

    #import itertools 
    #resultsIterator = itertools.imap(computeBound, paramList)
    
    i = 0    
    
    for i in range(folds):
        for j, lmbda in enumerate(maxLocalAuc.lmbdas): 
            trainObj, bound, testObj = resultsIterator.next() 
            trainObjs[j, i] = trainObj 
            bounds[j, i] = bound 
            testObjs[j, i] = testObj

    numpy.savez(outputFile, trainObjs, bounds, testObjs)
    
    pool.terminate()   
    logging.debug("Saved results in " + outputFile) 
    
    trainObjs = numpy.mean(trainObjs, 1)
    bounds = numpy.mean(bounds, 1)
    testObjs = numpy.mean(testObjs, 1)
    
else: 
    data = numpy.load(outputFile)
    trainObjs, bounds, testObjs = data["arr_0"], data["arr_1"], data["arr_2"]   
    import matplotlib.pyplot as plt   
    
    trainObjs = numpy.mean(trainObjs, 1)
    bounds = numpy.mean(bounds, 1)
    testObjs = numpy.mean(testObjs, 1)
    
    plotInds = ["k-", "k--", "k-.", "r-", "b-", "c-", "c--", "c-.", "g-", "g--", "g-."]    
    
    diffObjs = testObjs - trainObjs
    
    #Note we multiply by 2 since we have 1/2 for square and hinge losses and we need Q \in [0, 1]
    
    plt.figure(0)
    plt.plot(trainObjs*2, label="train obj")
    plt.plot(testObjs*2, label="test obj")
    plt.legend()    
    
    plt.figure(1)
    plt.plot(diffObjs*2, label="diff obj")
    plt.legend()
    
    plt.figure(2)
    plt.plot(bounds, label="bound")
    plt.legend()
    plt.show()

print(trainObjs)
print(testObjs)
print(bounds)