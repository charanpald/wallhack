import numpy
import logging
import sys
import multiprocessing 
import os
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from wallhack.rankingexp.DatasetUtils import DatasetUtils

"""
Test best starting point for average SGD  
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
expNum = 10

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
maxLocalAuc.t0 = 0.1
maxLocalAuc.lmbdaU = 0.0
maxLocalAuc.lmbdaV = 1.0
maxLocalAuc.rho = 0.5

maxItems = 10
chunkSize = 1
startAverages = numpy.array([2, 5, 10, 20, 30, 40])

learningRateParams = [(4.0, 1.0), (4.0, 0.5), (4.0, 0.1), (1.0, 1.0), (1.0, 0.5), (1.0, 0.1), (0.25, 1.0), (0.25, 0.5), (0.25, 0.1)]
print(startAverages)

def computeTestObj(args): 
    trainX, maxLocalAuc  = args 
    numpy.random.seed(21)
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(trainX, verbose=True)
    return U, V, trainMeasures[-1, 0], testMeasures[-1, 0]

if saveResults:    
    trainObjectives = numpy.zeros((startAverages.shape[0], len(learningRateParams)))
    testObjectives = numpy.zeros((startAverages.shape[0], len(learningRateParams)))
    
    for trainX, testX in trainTestXs: 
        trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
        testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
        allOmegaPtr = SparseUtils.getOmegaListPtr(X)
        logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))        
        
        paramList = []      
        
        for j, startAverage in enumerate(startAverages): 
            for i, (alpha, t0) in enumerate(learningRateParams):
                maxLocalAuc.startAverage = startAverage
                maxLocalAuc.alpha = alpha 
                maxLocalAuc.t0 = t0
                logging.debug(maxLocalAuc)
                
                learner = maxLocalAuc.copy()
                paramList.append((trainX, learner))
                
        pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
        resultsIterator = pool.imap(computeTestObj, paramList, chunkSize)
    
        for j, startAverage in enumerate(startAverages): 
            for i, (alpha, t0) in enumerate(learningRateParams):
                U, V, trainObj, testObj = resultsIterator.next()
                
                trainObjectives[j, i] += trainObj
                testObjectives[j, i] += testObj
        
        pool.terminate()        
        
    trainObjectives /= folds 
    testObjectives /= folds 
    
    numpy.savez(outputFile, trainObjectives, testObjectives)
else: 
    data = numpy.load(outputFile)
    trainObjectives, testObjectives = data["arr_0"], data["arr_1"]
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    
    for i, (alpha, t0) in enumerate(learningRateParams):
        plt.figure(i)
        plt.title("alpha=" + str(alpha) + "t0=" + str(t0))
        plt.plot(startAverages, trainObjectives[:, i], label="train")
        plt.plot(startAverages, testObjectives[:, i], label="test")
        plt.xlabel("startAverages")
        plt.ylabel("objs")
        plt.legend()

    plt.show()

print(trainObjectives)
print(testObjectives)

