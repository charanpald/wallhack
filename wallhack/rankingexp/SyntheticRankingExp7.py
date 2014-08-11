import numpy
import logging
import sys
import multiprocessing 
import os
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
from sandbox.util.Sampling import Sampling
from wallhack.rankingexp.DatasetUtils import DatasetUtils

"""
Test the effect of quantile threshold u and rank weight rho 
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
expNum = 7

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
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k, w2, eps=eps, stochastic=True)
maxLocalAuc.alpha = 4.0
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.folds = 2
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 1.0
maxLocalAuc.itemExpQ = 1.0
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.maxIterations = 100
maxLocalAuc.metric = "f1"
maxLocalAuc.normalise = True
maxLocalAuc.numAucSamples = 10
#maxLocalAuc.numProcesses = 1
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.rate = "optimal"
maxLocalAuc.recommendSize = 5
maxLocalAuc.recordStep = 10
maxLocalAuc.rho = 0.5
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationSize = 5

numRecordAucSamples = 200
maxItems = 10
chunkSize = 1
us = numpy.linspace(0, 1, 10)
rhos = numpy.linspace(-1, 2, 10)

def computeTestAuc(args): 
    trainX, maxLocalAuc  = args 
    numpy.random.seed(21)
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(trainX, verbose=True)
    return U, V, trainMeasures[-1, 1], testMeasures[-1, 1]

print(us)
print(rhos)

if saveResults: 
    trainLocalAucs = numpy.zeros((us.shape[0], rhos.shape[0]))
    trainPrecisions = numpy.zeros((us.shape[0], rhos.shape[0]))
    trainRecalls = numpy.zeros((us.shape[0], rhos.shape[0]))
    
    testLocalAucs = numpy.zeros((us.shape[0], rhos.shape[0]))
    testPrecisions = numpy.zeros((us.shape[0], rhos.shape[0]))
    testRecalls = numpy.zeros((us.shape[0], rhos.shape[0]))
    
    maxLocalAuc.learningRateSelect(X)    
    
    for trainX, testX in trainTestXs: 
        trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
        testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
        allOmegaPtr = SparseUtils.getOmegaListPtr(X)
        logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))        
        
        paramList = []      
        
        for i, u in enumerate(us): 
            maxLocalAuc.w = 1-u
            for j, rho in enumerate(rhos): 
                maxLocalAuc.rho = rho
                logging.debug(maxLocalAuc)
                
                learner = maxLocalAuc.copy()
                
                paramList.append((trainX, learner))

        pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
        resultsIterator = pool.imap(computeTestAuc, paramList, chunkSize)
        
        for i, u in enumerate(us): 
            for j, rho in enumerate(rhos): 
                U, V, trainAuc, testAuc = resultsIterator.next()
                trainLocalAucs[i, j] += trainAuc
                trainOrderedItems = MCEvaluator.recommendAtk(U, V, maxItems)    
                trainPrecisions[i, j] += MCEvaluator.precisionAtK(trainOmegaPtr, trainOrderedItems, maxItems)
                trainRecalls[i, j] += MCEvaluator.recallAtK(trainX, trainOrderedItems, maxItems)
                
                testLocalAucs[i, j] += testAuc
                testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, maxItems, trainX)
                testPrecisions[i, j] += MCEvaluator.precisionAtK(testX, testOrderedItems, maxItems)
                testRecalls[i, j] += MCEvaluator.recallAtK(testX, testOrderedItems, maxItems)
        
        pool.terminate()        
        
    testLocalAucs /= folds 
    testPrecisions /= folds 
    testRecalls /= folds 
    
    numpy.savez(outputFile, testLocalAucs, testPrecisions, testRecalls)
else: 
    data = numpy.load(outputFile)
    testLocalAucs, testPrecisions, testRecalls = data["arr_0"], data["arr_1"], data["arr_2"]
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
   
    plt.figure(0)
    plt.contourf(rhos, us, testLocalAucs)
    plt.xlabel("rho")
    plt.ylabel("u")
    plt.colorbar()
    
    plt.figure(1)
    plt.contourf(rhos, us, testPrecisions)
    plt.xlabel("rho")
    plt.ylabel("u")
    plt.colorbar()
    
    plt.figure(2)    
    plt.contourf(rhos, us, testRecalls)
    plt.xlabel("rho")
    plt.ylabel("u")
    plt.colorbar()   
   
    plt.show()

print(testLocalAucs)
print(testPrecisions)
print(testRecalls)
