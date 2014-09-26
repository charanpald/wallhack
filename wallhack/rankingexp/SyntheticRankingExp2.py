import numpy
import logging
import sys
import os
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from wallhack.rankingexp.DatasetUtils import DatasetUtils

"""
How much random sampling do we need for fast convergence 
"""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
saveResults = True
dataset = "synthetic" 
prefix = "Convergence"
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

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
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
maxLocalAuc.folds = 2
maxLocalAuc.initialAlg = "rand"
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

#The parameters arrays 
numRowSamplesArray = numpy.array([5, 10, 20, 50])
numAucSamplesArray = numpy.array([5, 10, 20, 50])
etas = numpy.array([0, 5, 10, 20])
startAverages = numpy.array([10, 20, 30])

maxLocalAuc.numRowSamples = 10
maxLocalAuc.numAucSamples = 10

for numRowSamples in numRowSamplesArray: 
    maxLocalAuc.numRowSamples = numRowSamples
    logging.debug(maxLocalAuc)
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, verbose=True)
    
    plt.figure(0)
    plt.plot(trainMeasures[:, 0], label="train nRow="+str(numRowSamples))

maxLocalAuc.numRowSamples = 10
maxLocalAuc.numAucSamples = 10

for numAucSamples in numAucSamplesArray: 
    maxLocalAuc.numAucSamples = numAucSamples
    logging.debug(maxLocalAuc)
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, verbose=True)

    plt.figure(1)
    plt.plot(trainMeasures[:, 0], label="train nAuc="+str(numAucSamples))
    

maxLocalAuc.numRowSamples = 10
maxLocalAuc.numAucSamples = 10

for eta in etas: 
    maxLocalAuc.eta = eta
    logging.debug(maxLocalAuc)
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, verbose=True)

    plt.figure(2)
    plt.plot(trainMeasures[:, 0], label="train eta="+str(eta))    

maxLocalAuc.numRowSamples = 10
maxLocalAuc.numAucSamples = 10
maxLocalAuc.eta = eta 

for startAverage in startAverages: 
    maxLocalAuc.startAverage = startAverage
    logging.debug(maxLocalAuc)
    U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, verbose=True)

    plt.figure(3)
    plt.plot(trainMeasures[:, 0], label="train startAverage="+str(startAverage))      
    
plt.figure(0)
plt.xlabel("iteration")
plt.ylabel("objective")
plt.legend()

plt.figure(1)
plt.xlabel("iteration")
plt.ylabel("objective")
plt.legend()

plt.figure(2)
plt.xlabel("iteration")
plt.ylabel("objective")
plt.legend()

plt.figure(3)
plt.xlabel("iteration")
plt.ylabel("objective")
plt.legend()

plt.show()
