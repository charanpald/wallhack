import numpy
import logging
import sys
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.Sampling import Sampling

"""
Look at the optimal learning rate parameters for values of k, lambda 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = False
expNum = 4

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

u = 0.1
w = 1.0 - u
k2 = 8
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k2, w, eps=eps, stochastic=True)
maxLocalAuc.maxIterations = 50
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.recordStep = 10
maxLocalAuc.rate = "optimal"
maxLocalAuc.rho = 0.5
maxLocalAuc.folds = 1

ks = numpy.array([8, 16, 32, 64])
lmbdas = numpy.array([0.5, 1.0, 2.0, 4.0, 8.0])

optimalAlphas = numpy.zeros((ks.shape[0],lmbdas.shape[0]))
optimalt0s = numpy.zeros((ks.shape[0], lmbdas.shape[0]))

if saveResults:
    
    for i, k in enumerate(ks): 
        for j, lmbda in enumerate(lmbdas):
            maxLocalAuc.k = k 
            maxLocalAuc.lmbdaV = lmbda             
            
            meanObjs = maxLocalAuc.learningRateSelect(X)
            
            optimalAlphas[i, j] = maxLocalAuc.alpha
            optimalt0s[i, j] = maxLocalAuc.t0


    numpy.savez(outputFile, optimalAlphas, optimalt0s)
else: 
    data = numpy.load(outputFile)
    optimalAlphas, optimalt0s = data["arr_0"], data["arr_1"]
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    
    print(optimalAlphas.shape)
    print(ks.shape)
    print(lmbdas.shape)
    
    plt.figure(0)
    plt.contourf(numpy.log2(lmbdas), numpy.log2(ks), optimalAlphas)
    plt.xlabel("lambda")
    plt.ylabel("k")
    plt.colorbar()
    
    plt.figure(1)
    plt.contourf(numpy.log2(lmbdas), numpy.log2(ks), optimalt0s)
    plt.xlabel("lambda")
    plt.ylabel("k")
    plt.colorbar()
    
    plt.show()
    
print(optimalAlphas)
print(optimalt0s)
