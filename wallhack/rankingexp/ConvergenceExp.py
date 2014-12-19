import numpy
import logging
import sys
import os
import multiprocessing
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from wallhack.rankingexp.DatasetUtils import DatasetUtils

"""
How much random sampling do we need for fast convergence 
"""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"


#Create a low rank matrix  
saveResults = True
prefix = "Convergence"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 
X = DatasetUtils.getDataset(dataset, nnz=20000)

m, n = X.shape 

folds = 3
testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, folds, testSize)
trainX, testX = trainTestXs[0]

trainOmegaList = SparseUtils.getOmegaList(trainX)
trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
testOmegaList = SparseUtils.getOmegaList(testX)
testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
allOmegaPtr = SparseUtils.getOmegaListPtr(X)
numRecordAucSamples = 200

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

#w = 1.0
k2 = 8
u2 = 10.0/n
w2 = 1-u2
eps = 10**-15
lmbda = 0.1
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.alpha = 0.1
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 6, 1)
maxLocalAuc.eta = 0
maxLocalAuc.folds = 2
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.lmbdas = 2.0**-numpy.arange(0, 10, 2)
maxLocalAuc.loss = "hinge"
maxLocalAuc.maxIterations = 50
maxLocalAuc.normalise = True
maxLocalAuc.numAucSamples = 10
#maxLocalAuc.numProcesses = 1
maxLocalAuc.numRecordAucSamples = 200
maxLocalAuc.numRowSamples = 10
maxLocalAuc.rate = "constant"
maxLocalAuc.recordStep = 5
maxLocalAuc.rho = 1.0
maxLocalAuc.t0 = 0.0001
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationUsers = 0.0

os.system('taskset -p 0xffffffff %d' % os.getpid())
chunkSize = 1

def computeObjectives(args): 
    trainX, maxLocalAuc, U, V  = args 
    numpy.random.seed(21)
    logging.debug(maxLocalAuc)
    
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, U=U, V=V, verbose=True)
    
    return trainMeasures[:, 0]

#The parameters arrays 
numRowSamplesArray = numpy.array([10, 20, 50])
numAucSamplesArray = numpy.array([5, 10, 20, 50])
etas = numpy.array([0, 5, 10, 20])
startAverages = numpy.array([0, 10, 20, 30])

if saveResults: 
    U, V = maxLocalAuc.initUV(trainX)    
    
    paramList = []
    objectives1 = numpy.zeros((numRowSamplesArray.shape[0], maxLocalAuc.maxIterations/maxLocalAuc.recordStep + 1, folds))
    
    for numRowSamples in numRowSamplesArray: 
        for trainX, testX in trainTestXs: 
            learner = maxLocalAuc.copy()
            learner.numRowSamples = numRowSamples
            paramList.append((trainX, learner, U, V))
    
    pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
    resultsIterator = pool.imap(computeObjectives, paramList, chunkSize)
    
    for i, numRowSamples in enumerate(numRowSamplesArray): 
        for j, (trainX, testX) in enumerate(trainTestXs): 
            objectives1[i, :, j]  = resultsIterator.next()
                
    pool.terminate()     
    
    
    paramList = []
    objectives2 = numpy.zeros((numAucSamplesArray.shape[0], maxLocalAuc.maxIterations/maxLocalAuc.recordStep + 1, folds))
    
    for numAucSamples in numAucSamplesArray: 
        for trainX, testX in trainTestXs: 
            learner = maxLocalAuc.copy()
            learner.numAucSamples = numAucSamples
            paramList.append((trainX, learner, U, V))
    
    pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
    resultsIterator = pool.imap(computeObjectives, paramList, chunkSize)
    
    for i, numAucSamples in enumerate(numAucSamplesArray): 
        for j, (trainX, testX) in enumerate(trainTestXs): 
            objectives2[i, :, j]  = resultsIterator.next()
                
    pool.terminate() 
    
    
    paramList = []
    objectives3 = numpy.zeros((etas.shape[0], maxLocalAuc.maxIterations/maxLocalAuc.recordStep + 1, folds))
    
    for eta in etas: 
        for trainX, testX in trainTestXs: 
            learner = maxLocalAuc.copy()
            learner.eta = eta
            paramList.append((trainX, learner, U, V))
    
    pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
    resultsIterator = pool.imap(computeObjectives, paramList, chunkSize)
    
    for i, eta in enumerate(etas): 
        for j, (trainX, testX) in enumerate(trainTestXs): 
            objectives3[i, :, j]  = resultsIterator.next()
                
    pool.terminate() 
  
    paramList = []
    objectives4 = numpy.zeros((startAverages.shape[0], maxLocalAuc.maxIterations/maxLocalAuc.recordStep + 1, folds))
    
    for startAverage in startAverages: 
        for trainX, testX in trainTestXs: 
            learner = maxLocalAuc.copy()
            learner.startAverage = startAverage
            paramList.append((trainX, learner, U, V))
    
    pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
    resultsIterator = pool.imap(computeObjectives, paramList, chunkSize)
    
    for i, startAverage in enumerate(startAverages): 
        for j, (trainX, testX) in enumerate(trainTestXs): 
            objectives4[i, :, j]  = resultsIterator.next()
                
    pool.terminate() 
    
    numpy.savez(outputFile, objectives1, objectives2, objectives3, objectives4)
    logging.debug("Saved results as " + outputFile)
else: 
    data = numpy.load(outputFile)
    objectives1, objectives2, objectives3, objectives4 = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]         
    
    objectivesMean1 = numpy.mean(objectives1, 2) 
    objectivesStd1 = numpy.std(objectives1, 2)

    objectivesMean2 = numpy.mean(objectives2, 2)
    objectivesStd2 = numpy.std(objectives2, 2)

    objectivesMean3 = numpy.mean(objectives3, 2)
    objectivesStd3 = numpy.std(objectives3, 2)
    
    objectivesMean4 = numpy.mean(objectives4, 2)
    objectivesStd4 = numpy.std(objectives4, 2)
    
    print(objectivesMean1)
    print(objectivesStd1)
    
    print(objectivesMean2)
    print(objectivesStd2)

    print(objectivesMean3)
    print(objectivesStd3)

    print(objectivesMean4)
    print(objectivesStd4)
    
    iterations = numpy.arange(0, maxLocalAuc.maxIterations+1, maxLocalAuc.recordStep)
    
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt     
    
    plt.figure(0)
    for i, numRowSamples in enumerate(numRowSamplesArray):     
        plt.plot(iterations, objectivesMean1[i, :], label="s_W = " + str(numRowSamples))
        
    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.legend()
    
    plt.figure(1)
    for i, numAucSamples in enumerate(numAucSamplesArray):     
        plt.plot(iterations, objectivesMean2[i, :], label="s_Y = " + str(numAucSamples))
    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.legend()
    
    plt.figure(2)
    for i, eta in enumerate(etas):     
        plt.plot(iterations, objectivesMean3[i, :], label="eta = " + str(eta))
    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.legend()
    
    plt.figure(3)
    for i, startAverage in enumerate(startAverages):     
        plt.plot(iterations, objectivesMean4[i, :], label="startAverage = " + str(startAverage))    
    
    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.legend()
    
    plt.show()
