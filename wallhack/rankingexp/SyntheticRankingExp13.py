
import numpy
import logging
import sys
import pickle 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.PathDefaults import PathDefaults 
from sandbox.util.Sampling import Sampling

"""
Look at ways to reduce random variability of algorithm. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#numpy.random.seed(22)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)

#Create a low rank matrix  
if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True

expNum = 13

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1(m=100, n=50)
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

k2 = 32
u2 = 0.1
w2 = 1-u2
eps = 10**-8
lmbda = 1.0
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=0.0, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.alpha = 1.0
maxLocalAuc.alphas = 2.0**-numpy.arange(-5, 5, 1)
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
maxLocalAuc.recordStep = 1
maxLocalAuc.rho = 0.0
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(-1, 6, 1)
maxLocalAuc.validationSize = 5
maxLocalAuc.validationUsers = 0 

t0s = 2.0**-numpy.arange(-2, 8, 1)

#maxLocalAuc.maxIterations = 1
#maxLocalAuc.numProcesses = 1

meanObjsList = []
stdObjsList = []

initialAlgs = ["rand", "svd"]
rates = ["optimal", "constant"]

if saveResults:
    #Run through all types of gradient descent to figure out which optimises the best 
    for stochastic in [False, True]: 
        for normalise in [False, True]: 
            for initialAlg in initialAlgs: 
                for rate in rates: 
                
                    maxLocalAuc.stochastic = stochastic
                    maxLocalAuc.initialAlg = initialAlg
                    
                    if initialAlg == "rand": 
                        maxLocalAuc.t0s = t0s
                    else: 
                        maxLocalAuc.t0s = t0s
                    
                    maxLocalAuc.normalise = normalise
                    maxLocalAuc.rate = rate
                        
                    meanObjs, stdObjs = maxLocalAuc.learningRateSelect(X)
                    
                    meanObjsList.append(meanObjs)
                    stdObjsList.append(stdObjs)
    
    pickle.dump((meanObjsList, stdObjsList), open(outputFile, "w"))
else: 
    data = numpy.load(outputFile)
    meanObjsList, stdObjsList = pickle.load(open(outputFile))
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    """
    plotInd = 0 
    for stochastic in [False, True]: 
        for normalise in [False, True]: 
            for initialAlg in initialAlgs: 
                print("stochastic=" + str(stochastic) + " normalise=" + str(normalise) + " initialAlg=" + str(initialAlg))
                meanObjs = meanObjsList[plotInd]
                stdObjs = stdObjsList[plotInd]
                
                if initialAlg == "rand": 
                    t0s = t0s2
                else: 
                    t0s = t0s1                
            
                plt.figure(plotInd)
                plt.contourf(numpy.log2(maxLocalAuc.alphas), numpy.log2(t0s), meanObjs)
                plt.xlabel("t0")
                plt.ylabel("alpha")
                plt.colorbar()
                plotInd += 1
                
                plt.figure(plotInd)
                plt.contourf(numpy.log2(maxLocalAuc.alphas), numpy.log2(t0s), stdObjs)
                plt.xlabel("t0")
                plt.ylabel("alpha")
                plt.colorbar()
        
        plotInd += 1
    """
    
    plt.show()
    

i = 0    

for stochastic in [False, True]: 
    for normalise in [False, True]: 
        for initialAlg in initialAlgs: 
            for rate in rates: 
                meanObjs = meanObjsList[i]
                stdObjs = stdObjsList[i]            
                
                print("stochastic=" + str(stochastic) + " normalise=" + str(normalise) + " initialAlg=" + str(initialAlg) + " rate=" + str(rate) + " min obj=" + str(numpy.min(meanObjs[numpy.isfinite(meanObjs)])))
    
                #print(meanObjs)
                #print(numpy.min(meanObjs[numpy.isfinite(meanObjs)]))
                #print(stdObjs)
                i += 1

#Results SVD results in lower objective and lower standard deviation 
