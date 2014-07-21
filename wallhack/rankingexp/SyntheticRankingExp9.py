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
Test best regularisation 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)
os.system('taskset -p 0xffffffff %d' % os.getpid())

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "movielens"

saveResults = True

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp9SyntheticResults.npz" 
elif dataset == "synthetic2": 
    X = DatasetUtils.syntheticDataset2()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp9Synthetic2Results.npz" 
elif dataset == "movielens": 
    X = DatasetUtils.movieLens()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp9MovieLensResults.npz" 
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp9FlixsterResults.npz" 
    Sampling.sampleUsers(X, 1000)
else: 
    raise ValueError("Unknown dataset: " + dataset)
        
testSize = 5
folds = 2
trainTestXs = Sampling.shuffleSplitRows(X, folds, testSize)

u = 0.1
w2 = 1-u 
k = 128
eps = 10**-8
maxLocalAuc = MaxLocalAUC(k, w2, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 20
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.5
maxLocalAuc.t0 = 0.1
maxLocalAuc.lmbdaU = 0.0
maxLocalAuc.rho = 0.5

maxItems = 10
chunkSize = 1
lmbdaVs = 10.0**numpy.arange(-0.5, 1.5, 0.25)
print(lmbdaVs)

def computeTestAuc(args): 
    trainX, maxLocalAuc  = args 
    numpy.random.seed(21)
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(trainX, verbose=True)
    return U, V, trainMeasures[-1, 1], testMeasures[-1, 1]

if saveResults:     
    testLocalAucs = numpy.zeros((lmbdaVs.shape[0]))
    testPrecisions = numpy.zeros((lmbdaVs.shape[0]))
    testRecalls = numpy.zeros((lmbdaVs.shape[0]))
    
    #maxLocalAuc.learningRateSelect(X)    
    
    for trainX, testX in trainTestXs: 
        trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
        testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
        allOmegaPtr = SparseUtils.getOmegaListPtr(X)
        logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))        
        
        paramList = []      
        
        for j, lmbdaV in enumerate(lmbdaVs): 
            maxLocalAuc.lmbdaV = lmbdaV
            logging.debug(maxLocalAuc)
            
            learner = maxLocalAuc.copy()
            paramList.append((trainX, learner))
                
        pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
        resultsIterator = pool.imap(computeTestAuc, paramList, chunkSize)
    
        for j, lmbdaV in enumerate(lmbdaVs): 
            U, V, trainAuc, testAuc = resultsIterator.next()
            
            testLocalAucs[j] += testAuc
            testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, maxItems, trainX)
            testPrecisions[j] += MCEvaluator.precisionAtK(testX, testOrderedItems, maxItems)
            testRecalls[j] += MCEvaluator.recallAtK(testX, testOrderedItems, maxItems)
        
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
    plt.plot(numpy.log10(lmbdaVs), testLocalAucs)
    plt.xlabel("lmbdaUs")
    plt.ylabel("AUC")
    
    plt.figure(1)
    plt.plot(numpy.log10(lmbdaVs), testPrecisions)
    plt.xlabel("lmbdaUs")
    plt.ylabel("precision")

    plt.figure(2)    
    plt.plot(numpy.log10(lmbdaVs), testRecalls)
    plt.xlabel("lmbdaUs")
    plt.ylabel("recall")
 
   
    plt.show()

print(testLocalAucs)
print(testPrecisions)
print(testRecalls)
