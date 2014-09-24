import numpy
import logging
import sys
import os
import multiprocessing
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils

from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from wallhack.rankingexp.DatasetUtils import DatasetUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
numpy.seterr(all="raise")

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True
prefix = "ROC"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 
print(outputFile)

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
elif dataset == "synthetic2": 
    X = DatasetUtils.syntheticDataset2()
elif dataset == "movielens": 
    X = DatasetUtils.movieLens(minNnzRows=20)
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    X = Sampling.sampleUsers(X, 1000)
else: 
    raise ValueError("Unknown dataset: " + dataset)

m, n = X.shape
u = 0.1 
w = 1-u

testSize = 5
folds = 3
trainTestXs = Sampling.shuffleSplitRows(X, folds, testSize)
trainX, testX = trainTestXs[0]

trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
allOmegaPtr = SparseUtils.getOmegaListPtr(X)
numRecordAucSamples = 200

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

k2 = 8
u2 = 0.1
w2 = 1-u2
eps = 10**-8
lmbda = 0.1
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=lmbda, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.maxIterations = 10
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 20
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.recordStep = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.rate = "constant"
maxLocalAuc.alpha = 0.5
maxLocalAuc.t0 = 1.0
maxLocalAuc.folds = 2
maxLocalAuc.rho = 1.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.validationSize = 3
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.normalise = True
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.metric = "f1"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.validationUsers = 0

os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
losses = ["tanh", "hinge", "square", "logistic"]

def computeTestAuc(args): 
    trainX, testX, maxLocalAuc  = args 
    numpy.random.seed(21)
    logging.debug(maxLocalAuc)
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, verbose=True)
    
    fprTrain, tprTrain = MCEvaluator.averageRocCurve(trainX, U, V)
    fprTest, tprTest = MCEvaluator.averageRocCurve(testX, U, V)
    
    return fprTrain, tprTrain, fprTest, tprTest

if saveResults: 
    paramList = []
    chunkSize = 1
    
    for loss in losses: 
        for trainX, testX in trainTestXs: 
            maxLocalAuc.loss = loss 
            
            paramList.append((trainX, testX, maxLocalAuc.copy()))
    
    pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
    resultsIterator = pool.imap(computeTestAuc, paramList, chunkSize)
    
    for loss in losses: 
        fprTrains = [] 
        tprTrains = [] 
        fprTests = [] 
        tprTests = []
        
        for trainX, testX in trainTestXs: 
            fprTrain, tprTrain, fprTest, tprTest = resultsIterator.next()
            
            fprTrains.append(fprTrain)
            tprTrains.append(tprTrain)
            fprTests.append(fprTest) 
            tprTests.append(tprTest)
            
        meanFprTrain = numpy.mean(numpy.array(fprTrains), 0)    
        meanTprTrain = numpy.mean(numpy.array(tprTrains), 0) 
        meanFprTest = numpy.mean(numpy.array(fprTests), 0) 
        meanTprTest = numpy.mean(numpy.array(tprTests), 0) 
        
        numpy.savez(outputFile, meanFprTrain, meanTprTrain, meanFprTest, meanTprTest)
    
    pool.terminate()   
else: 
    data = numpy.load(outputFile)
    meanFprTrain, meanTprTrain, meanFprTest, meanTprTest = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]      
   
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt    
   
    plt.figure(0)
    plt.plot(meanFprTrain, meanTprTrain, label=loss)
    
    plt.figure(1)    
    plt.plot(meanFprTest, meanTprTest, label=loss)    
    
    plt.figure(0)
    plt.xlabel("mean false positive rate")
    plt.ylabel("mean true positive rate")
    plt.legend()
    
    plt.figure(1)
    plt.xlabel("mean false positive rate")
    plt.ylabel("mean true positive rate")
    plt.legend()
    
    plt.show()