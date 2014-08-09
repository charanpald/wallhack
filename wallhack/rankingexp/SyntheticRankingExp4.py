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

saveResults = True
fixt0 = False

expNum = 4

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "SyntheticResults_" + str(fixt0) + ".npz" 
elif dataset == "synthetic2": 
    X = DatasetUtils.syntheticDataset2()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "Synthetic2Results_"  + str(fixt0) +  ".npz" 
elif dataset == "movielens": 
    X = DatasetUtils.movieLens()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "MovieLensResults_"  + str(fixt0) +  ".npz" 
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "FlixsterResults_"  + str(fixt0) +  ".npz"  
    X = Sampling.sampleUsers(X, 1000)
else: 
    raise ValueError("Unknown dataset: " + dataset)
    
m, n = X.shape

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

k2 = 128
u2 = 0.1
w2 = 1-u2
eps = 10**-8
lmbda = 1.0
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=0.0, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.alpha = 4.0
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.folds = 2
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
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationSize = 5

ks = numpy.array([8, 16, 32, 64, 128])
lmbdas = numpy.array([0.5, 1.0, 2.0, 4.0, 8.0])

optimalAlphas = numpy.zeros((ks.shape[0],lmbdas.shape[0]))
optimalt0s = numpy.zeros((ks.shape[0], lmbdas.shape[0]))
optimalObjs = numpy.zeros((ks.shape[0], lmbdas.shape[0])) 

if saveResults:
    
    for i, k in enumerate(ks): 
        for j, lmbda in enumerate(lmbdas):
            maxLocalAuc.k = k 
            maxLocalAuc.lmbdaV = lmbda             
            
            if fixt0:
                maxLocalAuc.t0s = numpy.array([lmbda])
                
            meanObjs = maxLocalAuc.learningRateSelect(X)
            
            optimalAlphas[i, j] = maxLocalAuc.alpha
            optimalt0s[i, j] = maxLocalAuc.t0
            optimalObjs[i, j] = numpy.min(meanObjs)

    numpy.savez(outputFile, optimalAlphas, optimalt0s, optimalObjs)
else: 
    data = numpy.load(outputFile)
    optimalAlphas, optimalt0s, optimalObjs = data["arr_0"], data["arr_1"], data["arr_2"]
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    
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
print(optimalObjs)
