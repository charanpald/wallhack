
import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.Sampling import Sampling

"""
Look at ways to reduce random variability of algorithm. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True

expNum = 13

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
    
m, n = X.shape

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

k2 = 64
u2 = 0.1
w2 = 1-u2
eps = 10**-8
lmbda = 1.0
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=0.0, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.alpha = 4.0
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.folds = 5
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 1.0
maxLocalAuc.itemExpQ = 1.0
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.maxIterations = 100
maxLocalAuc.metric = "f1"
maxLocalAuc.normalise = True
maxLocalAuc.numAucSamples = 10
#maxLocalAuc.numProcesses = 1
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.rate = "optimal"
maxLocalAuc.recommendSize = 5
maxLocalAuc.recordStep = 10
maxLocalAuc.rho = 0.5
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(-1, 6, 1)
maxLocalAuc.validationSize = 5


if saveResults:
    maxLocalAuc.t0s = 2.0**-numpy.arange(1, 8, 1)
    maxLocalAuc.initialAlg = "svd"
    meanObjs, stdObjs = maxLocalAuc.learningRateSelect(X)
    
    maxLocalAuc.t0s = 2.0**-numpy.arange(-1, 6, 1)
    maxLocalAuc.initialAlg = "rand"
    meanObjs2, stdObjs2 = maxLocalAuc.learningRateSelect(X)
    
    numpy.savez(outputFile, meanObjs, stdObjs, meanObjs2, stdObjs2)
else: 
    data = numpy.load(outputFile)
    meanObjs, stdObjs, meanObjs2, stdObjs2 = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    
    plt.figure(0)
    plt.contourf(numpy.log2(maxLocalAuc.t0s), numpy.log2(maxLocalAuc.alphas), meanObjs)
    plt.xlabel("t0")
    plt.ylabel("alpha")
    plt.colorbar()
    
    plt.figure(1)
    plt.contourf(numpy.log2(maxLocalAuc.t0s), numpy.log2(maxLocalAuc.alphas), meanObjs)
    plt.xlabel("t0")
    plt.ylabel("alpha")
    plt.colorbar()
    
    plt.show()
    
print(meanObjs)
print(stdObjs)

print(meanObjs2)
print(stdObjs2)
