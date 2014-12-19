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
We look at the ROC curves for the different objective functions 
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
#numpy.seterr(all="raise")

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True
prefix = "LossROC"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 
X = DatasetUtils.getDataset(dataset, nnz=20000)

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
eps = 10**-4
lmbda = 0.0
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=lmbda, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.alpha = 0.05
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.folds = 1
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.maxIterations = 500
maxLocalAuc.metric = "f1"
maxLocalAuc.normalise = True
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numProcesses = 1
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.rate = "constant"
maxLocalAuc.recordStep = 10
maxLocalAuc.rho = 1.0
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationSize = 3
maxLocalAuc.validationUsers = 0

os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
losses = [("tanh", 0.25), ("tanh", 0.5), ("tanh", 1.0), ("tanh", 2.0), ("hinge", 1), ("square", 1), ("logistic", 0.5), ("logistic", 1.0), ("logistic", 2.0), ("sigmoid", 0.5), ("sigmoid", 1.0), ("sigmoid", 2.0)]

def computeTestAuc(args): 
    trainX, testX, maxLocalAuc, U, V  = args 
    numpy.random.seed(21)
    logging.debug(maxLocalAuc)
    
    #maxLocalAuc.learningRateSelect(trainX)
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, U=U, V=V, verbose=True)
    
    fprTrain, tprTrain = MCEvaluator.averageRocCurve(trainX, U, V)
    fprTest, tprTest = MCEvaluator.averageRocCurve(testX, U, V)
        
    return fprTrain, tprTrain, fprTest, tprTest

if saveResults: 
    paramList = []
    chunkSize = 1
    
    U, V = maxLocalAuc.initUV(X)
    
    for loss, rho in losses: 
        for trainX, testX in trainTestXs: 
            maxLocalAuc.loss = loss 
            maxLocalAuc.rho = rho 
            paramList.append((trainX, testX, maxLocalAuc.copy(), U.copy(), V.copy()))

    pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
    resultsIterator = pool.imap(computeTestAuc, paramList, chunkSize)
    
    #import itertools 
    #resultsIterator = itertools.imap(computeTestAuc, paramList)
    
    meanFprTrains = []
    meanTprTrains = []
    meanFprTests = []
    meanTprTests = []
    
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
    
    #plotInds = ["k-", "k--", "k-.", "r-", "b-", "c-", "c--", "c-.", "g-", "g--", "g-."]
    plotInds = ["k-", "k--", "k-.", "k:", "r-"]
    ind = 0 
    
    #Figure out which losses to output   
    tanhMax = 0 
    sigmoidMax = 0 
    logisticMax = 0
    
    for i, lossTuple in enumerate(losses):
        loss, rho = lossTuple
        if loss == "tanh" and meanTprTrain[i, meanFprTrain[i, :]<=0.2][-1] > tanhMax:
            tanhMax = meanTprTrain[i, meanFprTrain[i, :]<=0.2][-1]
            tanhMaxRho = rho
        if loss == "sigmoid" and meanTprTrain[i, meanFprTrain[i, :]<=0.2][-1] > sigmoidMax:
            sigmoidMax = meanTprTrain[i, meanFprTrain[i, :]<=0.2][-1]
            sigmoidMaxRho = rho
        if loss == "logistic" and meanTprTrain[i, meanFprTrain[i, :]<=0.2][-1] > logisticMax:
            logisticMax = meanTprTrain[i, meanFprTrain[i, :]<=0.2][-1]
            logisticMaxRho = rho
    
    for i, lossTuple in enumerate(losses):
        loss, rho = lossTuple
        
        if loss == "tanh" and tanhMaxRho != rho: 
            continue
        if loss == "sigmoid" and sigmoidMaxRho != rho: 
            continue
        if loss == "logistic" and logisticMaxRho != rho: 
            continue
        
        if loss == "tanh": 
            label=loss + r" $\rho=$" + str(rho)
        elif loss == "sigmoid" or loss =="logistic": 
            label=loss + r" $\beta=$" + str(rho)
        else: 
            label = loss 
  
        fprTrainStart =   meanFprTrain[i, meanFprTrain[i, :]<=0.2]   
        tprTrainStart =   meanTprTrain[i, meanFprTrain[i, :]<=0.2]   
        
        plt.figure(0)
        plt.plot(fprTrainStart, tprTrainStart, plotInds[ind], label=label)
        
        plt.figure(1)
        plt.plot(meanFprTrain[i, :], meanTprTrain[i, :], plotInds[ind], label=label)
        
        fprTestStart =   meanFprTest[i, meanFprTest[i, :]<=0.2]   
        tprTestStart =   meanTprTest[i, meanFprTest[i, :]<=0.2]         
        
        plt.figure(2)    
        plt.plot(fprTestStart, tprTestStart, plotInds[ind], label=label)            
        
        plt.figure(3)    
        plt.plot(meanFprTest[i, :], meanTprTest[i, :], plotInds[ind], label=label)    
        
        ind += 1
    
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
