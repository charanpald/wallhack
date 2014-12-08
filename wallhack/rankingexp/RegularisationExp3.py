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
We look at the ROC curves on the test set using the regularisation parameters 
chosen on a training set versus full set.  
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
prefix = "Regularisation3"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 
X = DatasetUtils.getDataset(dataset)

m, n = X.shape
logging.debug(X.shape)
u = 0.1 
w = 1-u

testSize = 5
folds = 3
trainTestXs = Sampling.shuffleSplitRows(X, folds, testSize)

numRecordAucSamples = 200

k2 = 8
u2 = 0.5
w2 = 1-u2
eps = 10**-8
lmbda = 0.1
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=lmbda, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.alpha = 0.1
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.folds = 1
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.lmbdas = 2.0**-numpy.arange(0, 6)
maxLocalAuc.loss = "hinge"
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

losses = ["square", "hinge", "sigmoid", "logistic", "tanh"]
nnzs = [0.25, 1.0]

def computeTestAuc(args): 
    modelSelecX, trainX, testX, maxLocalAuc, U, V  = args 
    numpy.random.seed(21)
    logging.debug(maxLocalAuc)
    
    maxLocalAuc.modelSelect(modelSelecX)
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, U=U, V=V, verbose=True)
    
    fprTrain, tprTrain = MCEvaluator.averageRocCurve(trainX, U, V)
    fprTest, tprTest = MCEvaluator.averageRocCurve(testX, U, V)
        
    return fprTrain, tprTrain, fprTest, tprTest

if saveResults: 
    paramList = []
    chunkSize = 1
    
    U, V = maxLocalAuc.initUV(X)
    
    for loss in losses: 
        for nnz in nnzs: 
            for trainX, testX in trainTestXs: 
                numpy.random.seed(21)
                modelSelectX, userInds = Sampling.sampleUsers2(trainX, nnz*trainX.nnz)
                maxLocalAuc.loss = loss 
                paramList.append((modelSelectX, trainX, testX, maxLocalAuc.copy(), U.copy(), V.copy()))

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
        
        for nnz in nnzs:         
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
    
    i = 0
    ind = 0 
    plotInds = ["k-", "k--", "k-.", "r-", "b-", "c-", "c--", "c-.", "g-", "g--", "g-."]
    
    for i, loss in enumerate(losses):
        for j, nnz in enumerate(nnzs): 
       
            
            plt.figure(i*2)            
            fprTrainStart =   meanFprTrain[ind, meanFprTrain[ind, :]<=0.2]   
            tprTrainStart =   meanTprTrain[ind, meanFprTrain[ind, :]<=0.2]   
            plt.plot(fprTrainStart, tprTrainStart, plotInds[j*2], label="train nnz="+str(nnz))
            
            fprTestStart =   meanFprTest[ind, meanFprTest[ind, :]<=0.2]   
            tprTestStart =   meanTprTest[ind, meanFprTest[ind, :]<=0.2]         
            plt.plot(fprTestStart, tprTestStart, plotInds[j*2+1], label="test nnz="+str(nnz))              

            plt.title(loss)
            plt.xlabel("false positive rate")
            plt.ylabel("true positive rate")
            plt.legend(loc="lower right")
    
            plt.figure(i*2+1)
            plt.plot(meanFprTrain[ind, :], meanTprTrain[ind, :], plotInds[j*2], label="train nnz="+str(nnz)) 
            plt.plot(meanFprTest[ind, :], meanTprTest[ind, :], plotInds[j*2+1], label="test nnz="+str(nnz)) 
            plt.title(loss)
            plt.xlabel("false positive rate")
            plt.ylabel("true positive rate")
            plt.legend(loc="lower right")
            
            ind += 1
    
    
    plt.show()
