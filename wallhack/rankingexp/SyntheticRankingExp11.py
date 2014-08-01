import numpy
import logging
import sys
import multiprocessing 
import os
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
"""
Test parallel versus non-parallel SGD 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)
os.system('taskset -p 0xffffffff %d' % os.getpid())

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True
expNum = 11

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "SyntheticResults.npz" 
elif dataset == "synthetic2": 
    X = DatasetUtils.syntheticDataset2()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "Synthetic2Results.npz" 
elif dataset == "movielens": 
    X = DatasetUtils.movieLens()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "MovieLensResults.npz" 
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "FlixsterResults.npz"  
    X = Sampling.sampleUsers(X, 1000)
else: 
    raise ValueError("Unknown dataset: " + dataset)
        
testSize = 5
folds = 2
trainTestXs = Sampling.shuffleSplitRows(X, folds, testSize)

u = 0.1
w2 = 1-u 
k = 64
eps = 10**-8
maxLocalAuc = MaxLocalAUC(k, w2, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 10
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 1.0
maxLocalAuc.t0 = 0.5
maxLocalAuc.lmbdaU = 0.0
maxLocalAuc.lmbdaV = 1.0
maxLocalAuc.rho = 0.5
maxLocalAuc.validationSize = 0.0
maxLocalAuc.ks = numpy.array([8, 16])
maxLocalAuc.lmbdas = numpy.array([0.5, 1.0])
#maxLocalAuc.numProcesses = 1

maxItems = 10
chunkSize = 1


def computeTestObj(args): 
    trainX, testX, maxLocalAuc  = args 
    numpy.random.seed(21)
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(trainX, verbose=True)
    return U, V, trainMeasures[-1, 0], testMeasures[-1, 0]

if saveResults:    
    trainObjectives = numpy.zeros((2, maxLocalAuc.ks.shape[0], maxLocalAuc.lmbdas.shape[0]))
    testF1s = numpy.zeros((2, maxLocalAuc.ks.shape[0], maxLocalAuc.lmbdas.shape[0]))
    
    for trainX, testX in trainTestXs: 
        trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
        testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
        allOmegaPtr = SparseUtils.getOmegaListPtr(X)
        logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))        
        
        paramList = []      
        
        for i, k in enumerate(maxLocalAuc.ks): 
            for j, lmbdaV in enumerate(maxLocalAuc.lmbdas):
                maxLocalAuc.k = k 
                maxLocalAuc.lmbdaV = lmbdaV
                logging.debug(maxLocalAuc)
                
                learner = maxLocalAuc.copy()
                paramList.append((trainX, testX, learner))
                 
        pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
        resultsIterator = pool.imap(computeTestObj, paramList, chunkSize)
    
        for i, k in enumerate(maxLocalAuc.ks): 
            for j, lmbdaV in enumerate(maxLocalAuc.lmbdas):
                U, V, trainObj, testObj = resultsIterator.next()
                
                trainObjectives[0, i, j] += trainObj
                testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, maxItems, trainX)
                testF1s[0, i, j] += MCEvaluator.f1AtK(testX, testOrderedItems, maxItems)
        
        pool.terminate()  
        
        
        #Now learn using parallel SGD 
        maxLocalAuc.parallelSGD = True    
        maxLocalAuc.processes = 8
        print(maxLocalAuc.ks)
        print(maxLocalAuc.lmbdas)
        
        for i, k in enumerate(maxLocalAuc.ks): 
            for j, lmbdaV in enumerate(maxLocalAuc.lmbdas):
                maxLocalAuc.k = k 
                maxLocalAuc.lmbdaV = lmbdaV
                logging.debug(maxLocalAuc)
                
                U, V, iterations, totalTime = maxLocalAuc.learnModel(trainX, verbose=True)

                r = SparseUtilsCython.computeR(U, V, maxLocalAuc.w)
                trainObj = maxLocalAuc.objectiveApprox(trainOmegaPtr, U, V, r, maxLocalAuc.gi, maxLocalAuc.gp, maxLocalAuc.gq)
                
                trainObjectives[1, i, j] += trainObj
                testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, maxItems, trainX)
                testF1s[1, i, j] += MCEvaluator.f1AtK(testX, testOrderedItems, maxItems)
        
    trainObjectives /= folds 
    testF1s /= folds 
    
    numpy.savez(outputFile, trainObjectives, testF1s)
else: 
    data = numpy.load(outputFile)
    trainObjectives, testF1s = data["arr_0"], data["arr_1"]
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    
    plt.figure(0)
    plt.contourf(maxLocalAuc.ks, maxLocalAuc.lmbdas, trainObjectives[0, :, :])
    plt.xlabel("ks")
    plt.ylabel("lambdas")
    plt.colorbar()
    
    plt.figure(1)
    plt.contourf(maxLocalAuc.ks, maxLocalAuc.lmbdas, testF1s[0, :, :])
    plt.xlabel("ks")
    plt.ylabel("lambdas")
    plt.colorbar()    
    
    plt.figure(2)
    plt.contourf(maxLocalAuc.ks, maxLocalAuc.lmbdas, trainObjectives[1, :, :])
    plt.xlabel("ks")
    plt.ylabel("lambdas")
    plt.colorbar()
    
    plt.figure(3)
    plt.contourf(maxLocalAuc.ks, maxLocalAuc.lmbdas, testF1s[1, :, :])
    plt.xlabel("ks")
    plt.ylabel("lambdas")
    plt.colorbar()  

    plt.show()

print("\n")
print(trainObjectives[0, :, :])
print(trainObjectives[1, :, :])
print(testF1s[0, :, :])
print(testF1s[1, :, :])
