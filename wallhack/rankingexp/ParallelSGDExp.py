import numpy
import logging
import sys
import os
from scipy import interpolate 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.Sampling import Sampling 
from sandbox.util.Latex import Latex 

"""
Compare parallel versus non-parallel SGD 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
numpy.seterr(all="raise")


if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic2"

saveResults = True
prefix = "ParallelSGD"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + ".npz" 
X = DatasetUtils.getDataset(dataset, nnz=20000)

m, n = X.shape
u = 0.1 
w = 1-u


#w = 1.0
k = 8
u = 5/float(n)
w = 1-u
eps = 10**-12
lmbda = 10**-3
maxLocalAuc = MaxLocalAUC(k, w, eps=eps, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.alpha = 0.05
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.folds = 4
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([k])
maxLocalAuc.loss = "hinge"
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.lmbdaU = 0.1
maxLocalAuc.lmbdaV = 0.1
maxLocalAuc.maxIterations = 500
maxLocalAuc.metric = "f1"
maxLocalAuc.normalise = True
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numProcesses = 8
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.parallelSGD = True
maxLocalAuc.parallelStep = 1
maxLocalAuc.rate = "constant"
maxLocalAuc.recordStep = 5
maxLocalAuc.rho = 0.0
maxLocalAuc.startAverage = 30
maxLocalAuc.t0 = 0.1
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationSize = 3
maxLocalAuc.validationUsers = 0.0

os.system('taskset -p 0xffffffff %d' % os.getpid())

numCPUs = [2, 4, 8]
folds = 3

if saveResults: 
    logging.debug("Starting training")
    logging.debug(maxLocalAuc)
    
    
    objectivesArray = numpy.zeros((len(numCPUs)+1, maxLocalAuc.maxIterations/maxLocalAuc.recordStep+1, folds))
    timesArray = numpy.zeros((len(numCPUs)+1, maxLocalAuc.maxIterations/maxLocalAuc.recordStep+1, folds))
    iterationsArray = numpy.zeros((len(numCPUs)+1, maxLocalAuc.maxIterations/maxLocalAuc.recordStep+1, folds))
    
    for ind in range(folds): 
        #Non parallel version 
        maxLocalAuc.parallelSGD = False 
        initU, initV = maxLocalAuc.initUV(X)
        U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(X, U=initU, V=initV, verbose=True)  
        
        objectivesArray[0, :, ind] = trainMeasures[:, 0]
        timesArray[0, :, ind] = trainMeasures[:, 2]
        iterationsArray[0, :, ind] = trainMeasures[:, 3]
        
        #Second is parallel version 
        maxLocalAuc.parallelSGD = True 
        for i, j in enumerate(numCPUs): 
            initU, initV = maxLocalAuc.initUV(X)
            maxLocalAuc.numProcesses = j
            U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(X, U=initU, V=initV, verbose=True)  
            
            #Sometimes there are fewer iterations for parallel SGD 
            numSteps = trainMeasures.shape[0]
            objectivesArray[i+1, 0:numSteps, ind] = trainMeasures[:, 0]
            timesArray[i+1, 0:numSteps, ind] = trainMeasures[:, 2]
            iterationsArray[i+1, 0:numSteps, ind] = trainMeasures[:, 3]
        
    #objectivesArray = numpy.mean(objectivesArray, 2)
    #timesArray = numpy.mean(timesArray, 2)
    #iterationsArray = numpy.mean(iterationsArray, 2)

    numpy.savez(outputFile, objectivesArray, timesArray, iterationsArray)
    
    logging.debug("Saved results in " + outputFile) 
else: 
    data = numpy.load(outputFile)
    objectivesArray, timesArray, iterationsArray = data["arr_0"], data["arr_1"], data["arr_2"] 
    import matplotlib.pyplot as plt   

    newObjectivesArray = numpy.zeros(objectivesArray.shape)
    newTimesArray = numpy.zeros(timesArray.shape)
    newIterationsArray = numpy.zeros(iterationsArray.shape)

    #Have to use scipy interpolate here 
    for i in range(len(numCPUs)+1): 
        for j in range(folds):
            newIterationsArray[i, :, j] = numpy.arange(0, maxLocalAuc.maxIterations+1, maxLocalAuc.recordStep)            
            
            maxInd = numpy.argmax(iterationsArray[i, :, j])+1            
            
            f = interpolate.interp1d(iterationsArray[i, 0:maxInd, j], objectivesArray[i, 0:maxInd, j])
            newObjectivesArray[i, :, j] = f(newIterationsArray[i, :, j])
                        
            g = interpolate.interp1d(iterationsArray[i, 0:maxInd, j], timesArray[i, 0:maxInd, j])
            newTimesArray[i, :, j] = g(newIterationsArray[i, :, j])
    
    newIterationsArray = numpy.mean(newIterationsArray, 2)    
    newObjectivesArray = numpy.mean(newObjectivesArray, 2)
    newTimesArray = numpy.mean(newTimesArray, 2)
    
    #print(newIterationsArray[3, :])
    #print(newObjectivesArray[3, :])
    #print(newTimesArray[3, :])
    
    print(Latex.array1DToRow(newObjectivesArray[:, -1]))
    print(Latex.array1DToRow(numpy.max(newTimesArray, 1), precision=1))
    
    for i in range(len(numCPUs)+1): 
        if i==0: 
            label = "processes=1"
        else: 
            label = "processes=" + str(numCPUs[i-1]) 
        
        plotInds = ["k-", "k--", "k-.", "k:"]            
        
        maxInd = numpy.argmax(newIterationsArray[i, :])        
        
        plt.figure(0)
        plt.plot(newIterationsArray[i, 0:maxInd], newObjectivesArray[i, 0:maxInd], plotInds[i], label=label)
        plt.ylabel("objective")
        plt.xlabel("iteration")
        plt.legend()
    
    
        plt.figure(1)
        plt.plot(newTimesArray[i, 0:maxInd], newObjectivesArray[i, 0:maxInd], plotInds[i], label=label)
        plt.xlabel("time(s)")
        plt.ylabel("objective")
        plt.legend()
        
    plt.show()
