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
    dataset = "movielens"

#Create a low rank matrix  
saveResults = True
prefix = "Convergence2"
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

k2 = 64
u2 = 5/float(n)
w2 = 1-u2
eps = 10**-8
lmbda = 0.01
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=0.1, lmbdaV=0.1, stochastic=True)
maxLocalAuc.alpha = 32
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.beta = 2
maxLocalAuc.bound = False
maxLocalAuc.delta = 0.1
maxLocalAuc.eta = 0
maxLocalAuc.folds = 2
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([4, 8, 16, 32, 64, 128])
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.loss = "hinge" 
maxLocalAuc.maxIterations = 100
maxLocalAuc.maxNorm = 100
maxLocalAuc.metric = "f1"
maxLocalAuc.normalise = False
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numProcesses = 1
maxLocalAuc.numRecordAucSamples = 200
maxLocalAuc.numRowSamples = 30
maxLocalAuc.rate = "constant"
maxLocalAuc.recordStep = 10
maxLocalAuc.reg = False
maxLocalAuc.rho = 1.0
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationSize = 5
maxLocalAuc.validationUsers = 1.0

#Now try to get faster convergence 
t0s = numpy.array([0, 0.01, 0.1])
alphas = numpy.array([32, 64, 128, 256, 512])
etas = numpy.array([0, 10, 20, 50])
startAverages = numpy.array([10, 20, 50, 100])

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

print(objectives1)

inds = numpy.unravel_index(numpy.argmin(objectives1), objectives1.shape)

logging.debug("Small learning rate objective=" + str(numpy.min(idealTrainMeasures)))     
logging.debug("min obj=" + str(numpy.min(objectives1)))
logging.debug("t0=" + str(t0s[inds[0]]))
logging.debug("alpha=" + str(alphas[inds[1]]))
logging.debug("eta=" + str(etas[inds[2]]))
logging.debug("startAverage=" + str(startAverages[inds[3]]))
    
    