import numpy
import logging
import sys
import sppy
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
from sandbox.util.Sampling import Sampling
from wallhack.rankingexp.DatasetUtils import DatasetUtils

"""
Test the effect of varying rho. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "movielens"

saveResults = True

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp9SyntheticResults.npz" 
elif dataset == "movielens": 
    X = DatasetUtils.movieLens()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp9MovieLensResults.npz" 
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp9FlixsterResults.npz" 
    X = X[0:1000, :]
else: 
    raise ValueError("Unknown dataset: " + dataset)

m, n = X.shape

testSize = 5
folds = 3
trainTestXs = Sampling.shuffleSplitRows(X, folds, testSize)

u = 0.1 
w2 = 1-u 
k = 16
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k, w2, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 10
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.recordStep = 5
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0.5
maxLocalAuc.t0 = 10**-2
maxLocalAuc.lmbda = 0.01
maxLocalAuc.rho = 1.0

numRecordAucSamples = 200

maxItems = 5
rhos = numpy.linspace(0, 1, 5)

if saveResults: 
    trainLocalAucs = numpy.zeros(rhos.shape[0])
    trainPrecisions = numpy.zeros(rhos.shape[0])
    trainRecalls = numpy.zeros(rhos.shape[0])
    
    testLocalAucs = numpy.zeros(rhos.shape[0])
    testPrecisions = numpy.zeros(rhos.shape[0])
    testRecalls = numpy.zeros(rhos.shape[0])
    
    for trainX, testX in trainTestXs:     
        trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
        testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
        allOmegaPtr = SparseUtils.getOmegaListPtr(X)
        logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))        
    
        for i, rho in enumerate(rhos): 
            maxLocalAuc.rho = -rho
            logging.debug(maxLocalAuc)
            U, V, trainObjs, trainAucs, testObjs, testAucs, ind, totalTime = maxLocalAuc.learnModel(trainX, verbose=True)
            
            trainLocalAucs[i] += trainAucs[-1]
            trainOrderedItems = MCEvaluator.recommendAtk(U, V, maxItems)    
            trainPrecisions[i] += MCEvaluator.precisionAtK(trainOmegaPtr, trainOrderedItems, maxItems)
            trainRecalls[i] += MCEvaluator.recallAtK(trainX, trainOrderedItems, maxItems)
            
            testLocalAucs[i] += testAucs[-1]
            testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, maxItems, trainX)
            testPrecisions[i] += MCEvaluator.precisionAtK(testX, testOrderedItems, maxItems)
            testRecalls[i] += MCEvaluator.recallAtK(testX, testOrderedItems, maxItems)
        
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
    plt.plot(rhos, testLocalAucs)
    plt.xlabel("rho")
    plt.ylabel("local AUC")
    
    plt.figure(1)
    plt.plot(rhos, testPrecisions)
    plt.xlabel("rho")
    plt.ylabel("precision")
    
    plt.figure(2)
    plt.plot(rhos, testRecalls)
    plt.xlabel("rho")
    plt.ylabel("recall")
    
    plt.show()

print(testLocalAucs)
print(testPrecisions)
print(testRecalls)
