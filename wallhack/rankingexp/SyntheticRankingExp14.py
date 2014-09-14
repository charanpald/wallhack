import numpy
import logging
import sys
import os
import time 
import multiprocessing 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.PathDefaults import PathDefaults 

"""
Let's look at F1 scores as we go along for different top-n lists
"""


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
numpy.seterr(all="raise")


#X, U, V = DatasetUtils.syntheticDataset1()
X = DatasetUtils.syntheticDataset2()
#X = DatasetUtils.movieLens()

expNum = 14
outputFile = PathDefaults.getOutputDir() + "ranking/Exp" + str(expNum) + "SyntheticResults.npz" 

saveResults = False

m, n = X.shape
u = 0.1 
w = 1-u


#w = 1.0
k = 8
u = 1.0
w = 1-u
eps = 10**-8
lmbda = 10**-3
maxLocalAuc = MaxLocalAUC(k, w, eps=eps, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.maxIterations = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 20
maxLocalAuc.numRecordAucSamples = 200
maxLocalAuc.recordStep = 5
maxLocalAuc.parallelStep = 1
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.rate = "constant"
maxLocalAuc.alpha = 0.01
maxLocalAuc.lmbdaU = 0.01
maxLocalAuc.lmbdaV = 0.01
maxLocalAuc.ks = numpy.array([k])
maxLocalAuc.validationSize = 3
maxLocalAuc.validationUsers = 1.0
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.normalise = True
maxLocalAuc.numProcesses = 8
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.metric = "f1"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.itemFactors = False
maxLocalAuc.parallelSGD = False
maxLocalAuc.startAverage = 30
maxLocalAuc.recommendSize = numpy.array([1, 3, 5, 10])

os.system('taskset -p 0xffffffff %d' % os.getpid())


if saveResults: 
    logging.debug("Starting training")
    logging.debug(maxLocalAuc)
    
    numRuns = 5 
    chunkSize = 1
    numpy.random.seed(21)
    numRecords = maxLocalAuc.maxIterations/maxLocalAuc.recordStep + 1
    numPhis = 2  
    f1s = numpy.zeros((numRecords, maxLocalAuc.recommendSize.shape[0], numPhis, numRuns))
    aucs = numpy.zeros((numRecords, numPhis, numRuns)) 
    

    def computeTestMetricss(args): 
        X, maxLocalAuc  = args
        a = long(time.time() * 256)
        U, V, trainMeasures, testMeasures, iterations, totalTime = maxLocalAuc.learnModel(X, verbose=True, randSeed=a)
        return testMeasures[:, 1], testMeasures[:, 2:6]
    
    paramList = []
    
    for phi in [0, 1]: 
        for i in range(numRuns): 
            learner = maxLocalAuc.copy() 
            
            learner.phi = phi 
            paramList.append((X, learner))
            
    pool = multiprocessing.Pool(maxtasksperchild=100, processes=multiprocessing.cpu_count())
    resultsIterator = pool.imap(computeTestMetricss, paramList, chunkSize)
    
    for phi in [0, 1]: 
        for i in range(numRuns): 
            aucsArr, f1sArr = resultsIterator.next()
            
            aucs[:, phi, i] += aucsArr
            f1s[:, :, phi, i] += f1sArr
    
    pool.terminate()  
    

    numpy.savez(outputFile, f1s, aucs)
    
    logging.debug("Saved file as " + outputFile)
    
    meanF1s = f1s.mean(3)
    meanAucs = aucs.mean(2) 
    
    stdF1s = f1s.std(3)
    stdAucs = aucs.std(2)    
else: 
    data = numpy.load(outputFile)
    f1s, aucs = data["arr_0"], data["arr_1"]
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 

    meanF1s = f1s.mean(3)
    meanAucs = aucs.mean(2) 
    
    stdF1s = f1s.std(3)
    stdAucs = aucs.std(2)     

    iterations = numpy.arange(0, maxLocalAuc.maxIterations+1, maxLocalAuc.recordStep)

    plt.figure(0)
    plt.plot(iterations, meanAucs[:, 0], "k-", label="phi1")
    plt.plot(iterations, meanAucs[:, 1], "r-", label="phi2")
    plt.ylabel("AUC")
    plt.xlabel("iteration")
    plt.legend()
    

    plt.figure(1)
    #plt.plot(iterations, meanF1s[:, 0, 0], "k-", label="F1@" + str(maxLocalAuc.recommendSize[0]))
    plt.plot(iterations, meanF1s[:, 1, 1], "r-", label=r"$\phi_1$ F1@" + str(maxLocalAuc.recommendSize[1]))
    plt.plot(iterations, meanF1s[:, 2, 1], "r--", label=r"$\phi_1$ F1@" + str(maxLocalAuc.recommendSize[2]))
    #plt.plot(iterations, meanF1s[:, 3, 1], "r-.", label=r"$\phi_1$ F1@" + str(maxLocalAuc.recommendSize[3]))        
    plt.plot(iterations, meanF1s[:, 1, 0], "k-", label=r"$\phi_2$ F1@" + str(maxLocalAuc.recommendSize[1]))
    plt.plot(iterations, meanF1s[:, 2, 0], "k--", label=r"$\phi_2$ F1@" + str(maxLocalAuc.recommendSize[2]))
    #plt.plot(iterations, meanF1s[:, 3, 0], "k-.", label=r"$\phi_2$ F1@" + str(maxLocalAuc.recommendSize[3]))
    #plt.plot(iterations, meanF1s[:, 0, 1], "r-", label="F1@" + str(maxLocalAuc.recommendSize[0]))

    
    plt.ylabel("F1")
    plt.xlabel("iteration")
    plt.legend(loc="lower right")
    
    plt.show()
    
    
print(meanF1s[:, :, 0])
print(meanF1s[:, :, 1])
print(meanAucs)

print(stdF1s[:, :, 0])
print(stdF1s[:, :, 1])
print(stdAucs)