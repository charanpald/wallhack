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

"""
We look at the ROC curves for tanh with different rho parameters. 
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
numpy.seterr(all="raise")

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True
prefix = "Tanh"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 
print(outputFile)

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
elif dataset == "synthetic2": 
    X = DatasetUtils.syntheticDataset2()
elif dataset == "movielens": 
    X = DatasetUtils.movieLens()
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    X, userInds = Sampling.sampleUsers2(X, 50000)
else: 
    raise ValueError("Unknown dataset: " + dataset)

m, n = X.shape
u = 0.1 
w = 1-u

testSize = 5
folds = 5
trainTestXs = Sampling.shuffleSplitRows(X, folds, testSize)

numRecordAucSamples = 200

k2 = 8
u2 = 0.5
w2 = 1-u2
eps = 10**-8
lmbda = 0.1
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=lmbda, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.maxIterations = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.recordStep = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.rate = "constant"
maxLocalAuc.alpha = 0.1
maxLocalAuc.t0 = 1.0
maxLocalAuc.folds = 1
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
maxLocalAuc.numProcesses = 1
maxLocalAuc.loss = "tanh"

os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
rhos = numpy.array([0.5, 1.0, 2.0, 5.0, 10, 20])

def computeTestAuc(args): 
    trainX, testX, maxLocalAuc, U, V  = args 
    numpy.random.seed(21)
    logging.debug(maxLocalAuc)
    
    maxLocalAuc.learningRateSelect(trainX)
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, U=U, V=V, verbose=True)
    
    fprTrain, tprTrain = MCEvaluator.averageRocCurve(trainX, U, V)
    fprTest, tprTest = MCEvaluator.averageRocCurve(testX, U, V)
        
    return fprTrain, tprTrain, fprTest, tprTest

if saveResults: 
    paramList = []
    chunkSize = 1
    
    U, V = maxLocalAuc.initUV(X)
    
    for rho in rhos: 
        for trainX, testX in trainTestXs: 
            maxLocalAuc.rho = rho 
            paramList.append((trainX, testX, maxLocalAuc.copy(), U, V))

    pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
    resultsIterator = pool.imap(computeTestAuc, paramList, chunkSize)
    
    #import itertools 
    #resultsIterator = itertools.imap(computeTestAuc, paramList)
    
    meanFprTrains = []
    meanTprTrains = []
    meanFprTests = []
    meanTprTests = []
    
    for rho in rhos: 
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
        
        meanFprTrains.append(meanFprTrain)
        meanTprTrains.append(meanTprTrain)
        meanFprTests.append(meanFprTest)
        meanTprTests.append(meanTprTest)
        
    numpy.savez(outputFile, meanFprTrains, meanTprTrains, meanFprTests, meanTprTests)
    
    pool.terminate()   
    logging.debug("Saved results in " + outputFile)
else: 
    data = numpy.load(outputFile)
    meanFprTrain, meanTprTrain, meanFprTest, meanTprTest = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]      
   
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt   
    
    plotInds = ["k-", "k--", "k-.", "r-", "b-", "c-", "c--", "c-.", "g-", "g--", "g-."]
    
    for i, rho in enumerate(rhos):

        label= r" $\rho=$" + str(rho)

        
        fprTrainStart =   meanFprTrain[i, meanFprTrain[i, :]<=0.2]   
        tprTrainStart =   meanTprTrain[i, meanFprTrain[i, :]<=0.2]   
        
        plt.figure(0)
        plt.plot(fprTrainStart, tprTrainStart, plotInds[i], label=label)
        
        plt.figure(1)
        plt.plot(meanFprTrain[i, :], meanTprTrain[i, :], plotInds[i], label=label)
        
        fprTestStart =   meanFprTest[i, meanFprTest[i, :]<=0.2]   
        tprTestStart =   meanTprTest[i, meanFprTest[i, :]<=0.2]         
        
        plt.figure(2)    
        plt.plot(fprTestStart, tprTestStart, plotInds[i], label=label)            
        
        plt.figure(3)    
        plt.plot(meanFprTest[i, :], meanTprTest[i, :], plotInds[i], label=label)    
    
    plt.figure(0)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="lower right")
    
    plt.figure(1)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="lower right")
    
    plt.figure(2)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="lower right")
    
    plt.figure(3)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="lower right")
    
    plt.show()