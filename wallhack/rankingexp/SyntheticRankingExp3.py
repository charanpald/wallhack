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
Test the effect of item exponents 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)
os.system('taskset -p 0xffffffff %d' % os.getpid())

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "movielens"

saveResults = False

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp3SyntheticResults.npz" 
elif dataset == "synthetic2": 
    X = DatasetUtils.syntheticDataset2()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp3Synthetic2Results.npz" 
elif dataset == "movielens": 
    X = DatasetUtils.movieLens()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp3MovieLensResults.npz" 
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp3FlixsterResults.npz" 
    X = Sampling.sampleUsers(X, 1000)
else: 
    raise ValueError("Unknown dataset: " + dataset)

m, n = X.shape        
testSize = 5
itemExp = 0.5
folds = 2
colProbs = (X.sum(0)+1)/float(m+1)
colProbs = colProbs**-itemExp 
trainTestXs = Sampling.shuffleSplitRows(X, folds, testSize)

#w = 1.0
k2 = 128
u2 = 0.1
w2 = 1-u2
eps = 10**-8
lmbda = 1.0
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=0.0, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.alpha = 4.0
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.folds = 2
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 1.0
maxLocalAuc.itemExpQ = 1.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.maxIterations = 100
maxLocalAuc.metric = "f1"
maxLocalAuc.normalise = True
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numProcesses = 1
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.rate = "optimal"
maxLocalAuc.recommendSize = 5
maxLocalAuc.recordStep = 10
maxLocalAuc.rho = 0.5
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationSize = 5

maxItems = 3
chunkSize = 1
itemExpPs = numpy.linspace(0, 2, 6)
itemExpQs = numpy.linspace(0, 2, 6)

def computeTestAuc(args): 
    trainX, maxLocalAuc  = args 
    numpy.random.seed(21)
    U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(trainX, verbose=True)
    return U, V, trainMeasures[-1, 1], testMeasures[-1, 1]

if saveResults: 
    testPrecisions = numpy.zeros((itemExpPs.shape[0], itemExpQs.shape[0]))
    testRecalls = numpy.zeros((itemExpPs.shape[0], itemExpQs.shape[0]))
    
    #maxLocalAuc.learningRateSelect(X)    
    
    for trainX, testX in trainTestXs: 
        trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
        testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
        allOmegaPtr = SparseUtils.getOmegaListPtr(X)
        logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))        
        
        paramList = []      
        
        for i, itemExpP in enumerate(itemExpPs): 
            maxLocalAuc.itemExpP = itemExpP
            for j, itemExpQ in enumerate(itemExpQs): 
                maxLocalAuc.itemExpQ = itemExpQ
                
                learner = maxLocalAuc.copy()
                logging.debug(learner)

                paramList.append((trainX, learner))

        pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
        resultsIterator = pool.imap(computeTestAuc, paramList, chunkSize)
        
        for i, u in enumerate(itemExpPs): 
            for j, rho in enumerate(itemExpQs): 
                U, V, trainAuc, testAuc = resultsIterator.next()
                
                testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, maxItems, trainX)
                testPrecisions[i, j] += MCEvaluator.precisionAtK(testX, testOrderedItems, maxItems)
                testRecalls[i, j] += MCEvaluator.recallAtK(testX, testOrderedItems, maxItems)
        
        pool.terminate()        
        
    testPrecisions /= folds 
    testRecalls /= folds 
    
    numpy.savez(outputFile, testPrecisions, testRecalls)
else: 
    data = numpy.load(outputFile)
    testPrecisions, testRecalls = data["arr_0"], data["arr_1"]
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
   
    plt.figure(1)
    plt.contourf(itemExpPs, itemExpQs, testPrecisions)
    plt.xlabel("itemExpPs")
    plt.ylabel("itemExpQs")
    plt.colorbar()
    
    plt.figure(2)    
    plt.contourf(itemExpPs, itemExpQs, testRecalls)
    plt.xlabel("itemExpPs")
    plt.ylabel("itemExpQs")
    plt.colorbar()   
   
    plt.show()

print(testPrecisions)
print(testRecalls)

#On sythetic best is 1.0, 1.0 
# synthetic2 best is 2.0, 2.0 
#MovieLens best is 1.5, 1.0