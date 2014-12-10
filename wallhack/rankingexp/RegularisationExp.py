import numpy
import logging
import sys
import os
import multiprocessing
import itertools 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.Util import Util 
Util.setupScript()

"""
We look at the ROC curves on the test set for different values of maxNorm. We want 
to find why on epinions, the learning overfits so vary lambdaU and lambdaV  
"""

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "epinions"

saveResults = True
prefix = "Regularisation"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 
X = DatasetUtils.getDataset(dataset)

m, n = X.shape
u = 0.1 
w = 1-u

logging.debug("Sampled X shape: " + str(X.shape))

testSize = 5
folds = 5
trainTestXs = Sampling.shuffleSplitRows(X, folds, testSize)

numRecordAucSamples = 200

k2 = 64
u2 = 0.5
w2 = 1-u2
eps = 10**-8
lmbda = 0.0
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=lmbda, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.alpha = 0.05
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 9, 1)
maxLocalAuc.folds = 1
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.lmbdas = 2.0**-numpy.arange(-5, 6, 3)
maxLocalAuc.loss = "hinge"
maxLocalAuc.maxIterations = 500
maxLocalAuc.maxNorm = 1
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

softImpute = IterativeSoftImpute(k=k2, postProcess=True)

maxNorms = 2.0**numpy.arange(-3, 4)

#numProcesses = 1
numProcesses = multiprocessing.cpu_count()
os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")

def computeTestAuc(args): 
    trainX, testX, maxLocalAuc, U, V  = args 
    numpy.random.seed(21)
    logging.debug(maxLocalAuc)
    
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, U=U, V=V, verbose=True)
    
    fprTrain, tprTrain = MCEvaluator.averageRocCurve(trainX, U, V)
    fprTest, tprTest = MCEvaluator.averageRocCurve(testX, U, V, trainX=trainX)
        
    return fprTrain, tprTrain, fprTest, tprTest

if saveResults: 
    paramList = []
    chunkSize = 1
    
    #First generate SoftImpute results as a benchmark. 
    trainX, testX = trainTestXs[0]
    learner = softImpute.copy()
    
    trainIterator = iter([trainX.toScipyCsc()])
    ZList = learner.learnModel(trainIterator)    
    U, s, V = ZList.next()
    U = U*s
    
    U = numpy.ascontiguousarray(U)
    V = numpy.ascontiguousarray(V)
    
    fprTrainSI, tprTrainSI = MCEvaluator.averageRocCurve(trainX, U, V)
    fprTestSI, tprTestSI = MCEvaluator.averageRocCurve(testX, U, V, trainX=trainX)
    
    #Now train MaxLocalAUC 
    U, V = maxLocalAuc.initUV(X)
    
    for maxNorm in maxNorms:  
        for trainX, testX in trainTestXs: 
            learner = maxLocalAuc.copy()
            learner.maxNorm = maxNorm 
            paramList.append((trainX, testX, learner, U.copy(), V.copy()))

    if numProcesses != 1: 
        pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
        resultsIterator = pool.imap(computeTestAuc, paramList, chunkSize)
    else: 
        resultsIterator = itertools.imap(computeTestAuc, paramList)
    
    meanFprTrains = []
    meanTprTrains = []
    meanFprTests = []
    meanTprTests = []
    
    for maxNorm in maxNorms:  
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
        
    numpy.savez(outputFile, meanFprTrains, meanTprTrains, meanFprTests, meanTprTests, fprTrainSI, tprTrainSI, fprTestSI, tprTestSI)
    
    if numProcesses != 1: 
        pool.terminate()   
    logging.debug("Saved results in " + outputFile)
else: 
    data = numpy.load(outputFile)
    meanFprTrain, meanTprTrain, meanFprTest, meanTprTest, fprTrainSI, tprTrainSI, fprTestSI, tprTestSI = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"], data["arr_5"], data["arr_6"], data["arr_7"]      
   
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt   
    
    #print(meanFprTrain[0, :])
    #print(meanTprTrain[0, :])
    
    plotInds = ["k-", "k--", "k-.", "k:", "r-", "r--", "r-.", "r:", "g-", "g--", "g-.", "g:", "b-", "b--", "b-.", "b:", "c-"]
    ind = 0 
    
    for i, maxNorm in enumerate(maxNorms):
        label = r"$maxNorm=$" + str(maxNorm)

        fprTrainStart =   meanFprTrain[ind, meanFprTrain[ind, :]<=0.2]   
        tprTrainStart =   meanTprTrain[ind, meanFprTrain[ind, :]<=0.2]
        
        print(fprTrainStart, tprTrainStart)
        
        plt.figure(0)
        plt.plot(fprTrainStart, tprTrainStart, plotInds[ind], label=label)
        
        plt.figure(1)
        plt.plot(meanFprTrain[ind, :], meanTprTrain[ind, :], plotInds[ind], label=label)
        
        fprTestStart =   meanFprTest[ind, meanFprTest[ind, :]<=0.2]   
        tprTestStart =   meanTprTest[ind, meanFprTest[ind, :]<=0.2]         
        
        plt.figure(2)    
        plt.plot(fprTestStart, tprTestStart, plotInds[ind], label=label)            
        
        plt.figure(3)    
        plt.plot(meanFprTest[ind, :], meanTprTest[ind, :], plotInds[ind], label=label)    
        
        ind += 1
    
    plt.figure(1)
    plt.plot(fprTrainSI, tprTrainSI, plotInds[ind], label="SI")    
    
    plt.figure(3)
    plt.plot(fprTestSI, tprTestSI , plotInds[ind], label="SI")       
    
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


#Results are best with maxNorm=0.5 or 1.0 (lower might improve further)
#SI is still best on Epinions, Synthetic 