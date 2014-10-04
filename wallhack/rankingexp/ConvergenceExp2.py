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
Look at averaged stochastic gradient descent and parameters to get best 
convergence. 
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
prefix = "Convergence2"
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
maxLocalAuc.alpha = 0.001
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 6, 1)
maxLocalAuc.eta = 0
maxLocalAuc.folds = 2
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.lmbdas = 2.0**-numpy.arange(0, 10, 2)
maxLocalAuc.loss = "square"
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

#Now try to get faster convergence 
t0s = numpy.array([0, 0.5, 1.0])
alphas = numpy.array([1.0, 0.5, 0.25])
etas = numpy.array([0, 5, 10])
startAverages = numpy.array([5, 10, 20, 50])

os.system('taskset -p 0xffffffff %d' % os.getpid())
chunkSize = 1



def computeObjectives(args): 
    trainX, maxLocalAuc, U, V  = args 
    numpy.random.seed(21)
    logging.debug(maxLocalAuc)
    
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, U=U, V=V, verbose=True)
    
    return trainMeasures[-1, 0]
    
    
if saveResults: 
    #First run with low learning rate to get a near-optimal solution 
    U, V = maxLocalAuc.initUV(trainX)  
    maxLocalAuc.maxIterations = 2000
    U2, V2, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, U=U, V=V, verbose=True)
    
    idealTrainMeasures = trainMeasures[:, 0]    
    
    maxLocalAuc.maxIterations = 100
    
    paramList = []
    objectives1 = numpy.zeros((t0s.shape[0], alphas.shape[0], etas.shape[0], startAverages.shape[0], folds))
    
    for t0 in t0s: 
        for alpha in alphas: 
            for eta in etas: 
                for startAverage in startAverages: 
                    for trainX, testX in trainTestXs: 
                        learner = maxLocalAuc.copy()
                        learner.t0 = t0
                        learner.alpha = alpha 
                        learner.eta = eta 
                        learner.startAverage = startAverage
                        paramList.append((trainX, learner, U.copy(), V.copy()))
    
    pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
    resultsIterator = pool.imap(computeObjectives, paramList, chunkSize)
    
    for i, t0 in enumerate(t0s): 
        for j, alpha in enumerate(alphas): 
            for s, eta in enumerate(etas): 
                for t, startAverage in enumerate(startAverages): 
                    for u, (trainX, testX) in enumerate(trainTestXs): 
                        objectives1[i, j, s, t, u]  = resultsIterator.next()
                
    pool.terminate()     
    
    numpy.savez(outputFile, objectives1, idealTrainMeasures)
    logging.debug("Saved results as " + outputFile)
    
else: 
    data = numpy.load(outputFile)
    objectives1, idealTrainMeasures = data["arr_0"],  data["arr_1"]
    
objectives1 = numpy.mean(objectives1, 4)
inds = numpy.unravel_index(numpy.argmin(objectives1), objectives1.shape)

logging.debug("Small learning rate objective=" + str(numpy.min(idealTrainMeasures)))     
logging.debug("min obj=" + str(numpy.min(objectives1)))
logging.debug("t0=" + str(t0s[inds[0]]))
logging.debug("alpha=" + str(alphas[inds[1]]))
logging.debug("eta=" + str(etas[inds[2]]))
logging.debug("startAverage=" + str(startAverages[inds[3]]))
    
    