import numpy
import logging
import sys
import os
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling 
from wallhack.rankingexp.DatasetUtils import DatasetUtils

"""
Let's see if we can get the right learning rate on a subsample of rows 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
#numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

os.system('taskset -p 0xffffffff %d' % os.getpid())

#Create a low rank matrix  
if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True
expNum = 6

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "SyntheticResults.npz" 
elif dataset == "synthetic2": 
    X = DatasetUtils.syntheticDataset2()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "Synthetic2Results.npz" 
elif dataset == "movielens": 
    X = DatasetUtils.movieLens()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "MovieLensResults.npz" 
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "FlixsterResults.npz"  
    X = Sampling.sampleUsers(X, 1000)
else: 
    raise ValueError("Unknown dataset: " + dataset)

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
#logging.debug("Total local AUC:" + str(MCEvaluator.localAUC(X, U, V, w)))
#logging.debug("Train local AUC:" + str(MCEvaluator.localAUC(trainX, U, V, w)))
#logging.debug("Test local AUC:" + str(MCEvaluator.localAUC(testX, U, V, w)))

u = 0.1
w = 1-u
k2 = 16
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k2, w, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 10
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 1.0
maxLocalAuc.t0 = 0.1
maxLocalAuc.lmbdaU = 0.0
maxLocalAuc.lmbdaV = 1.0
#maxLocalAuc.numProcesses = 1
maxLocalAuc.t0s = 2.0**-numpy.arange(0, 6)
maxLocalAuc.alphas = 2.0**-numpy.arange(-1,3.5, 0.5)
maxLocalAuc.folds = 3

newM = trainX.shape[0]/2
modelSelectX = Sampling.sampleUsers(X, newM)



if saveResults: 
    meanObjs1 = maxLocalAuc.learningRateSelect(X)
    meanObjs2 = maxLocalAuc.learningRateSelect(trainX)
    meanObjs3 = maxLocalAuc.learningRateSelect(modelSelectX)

    numpy.savez(outputFile, meanObjs1, meanObjs2, meanObjs3)
else: 
    data = numpy.load(outputFile)
    meanObjs1, meanObjs2, meanObjs3 = data["arr_0"], data["arr_1"], data["arr_2"]
    
    import matplotlib.pyplot as plt 
    
    print(maxLocalAuc.t0s)
    print(maxLocalAuc.alphas)

    plt.figure(0)
    plt.contourf(numpy.log2(maxLocalAuc.t0s), numpy.log2(maxLocalAuc.alphas), meanObjs1)
    plt.xlabel("t0")
    plt.ylabel("alpha")
    plt.colorbar()
    
    plt.figure(1)
    plt.contourf(numpy.log2(maxLocalAuc.t0s), numpy.log2(maxLocalAuc.alphas), meanObjs2)
    plt.xlabel("t0")
    plt.ylabel("alpha")
    plt.colorbar()
    
    plt.figure(2)    
    plt.contourf(numpy.log2(maxLocalAuc.t0s), numpy.log2(maxLocalAuc.alphas), meanObjs3)
    plt.xlabel("t0")
    plt.ylabel("alpha")
    plt.colorbar()
    
    plt.show()

print(meanObjs1)
print(meanObjs2)
print(meanObjs3)
