import numpy
import logging
import sys
import multiprocessing 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.Util import Util 
Util.setupScript()

"""
Script to see if the learning is the same on a subset of rows or elements. 
"""

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "flixster"

saveResults = True
prefix = "LearningRate"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 

k2 = 64
w2 = 0.9
eps = 10**-8
lmbda = 0.01
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=0.1, lmbdaV=0.1, stochastic=True)
maxLocalAuc.alpha = 50
maxLocalAuc.alphas = 2.0**numpy.arange(3, 10)
maxLocalAuc.beta = 2
maxLocalAuc.bound = False
maxLocalAuc.delta = 0.1
maxLocalAuc.eta = 0
maxLocalAuc.folds = 2
maxLocalAuc.initialAlg = "svd"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([4, 8, 16, 32, 64, 128])
maxLocalAuc.lmbdas = 2.0**-numpy.arange(1, 7)
maxLocalAuc.loss = "hinge" 
maxLocalAuc.maxIterations = 500
maxLocalAuc.maxNorm = 100
maxLocalAuc.metric = "f1"
maxLocalAuc.normalise = False
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numProcesses = multiprocessing.cpu_count()
maxLocalAuc.numRecordAucSamples = 200
maxLocalAuc.numRowSamples = 30
maxLocalAuc.rate = "constant"
maxLocalAuc.recordStep = 10
maxLocalAuc.reg = False
maxLocalAuc.rho = 1.0
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationSize = 5
maxLocalAuc.validationUsers = 0.0

if saveResults: 
    X = DatasetUtils.getDataset(dataset, nnz=1000000)
    X2, userInds = Sampling.sampleUsers2(X, 500000, prune=True)
    X3, userInds = Sampling.sampleUsers2(X, 200000, prune=True)
    X4, userInds = Sampling.sampleUsers2(X, 100000, prune=True)
    X5, userInds = Sampling.sampleUsers2(X, 50000, prune=True)
    
    print(X.shape, X.nnz)
    print(X2.shape, X2.nnz)  
    print(X3.shape, X3.nnz)  
    print(X4.shape, X4.nnz)  
    print(X5.shape, X5.nnz)   
    
    meanObjs1, paramDict = maxLocalAuc.learningRateSelect(X)
    meanObjs2, paramDict = maxLocalAuc.learningRateSelect(X2)
    meanObjs3, paramDict = maxLocalAuc.learningRateSelect(X3)
    meanObjs4, paramDict = maxLocalAuc.learningRateSelect(X4)
    meanObjs5, paramDict = maxLocalAuc.learningRateSelect(X5)

    numpy.savez(outputFile, meanObjs1, meanObjs2, meanObjs3, meanObjs4, meanObjs5)
else: 
    data = numpy.load(outputFile)
    meanObjs1, meanObjs2, meanObjs3, meanObjs4, meanObjs5 = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"]
    
    meanObjs1 = numpy.squeeze(meanObjs1)    
    meanObjs2 = numpy.squeeze(meanObjs2) 
    meanObjs3 = numpy.squeeze(meanObjs3)
    meanObjs4 = numpy.squeeze(meanObjs4)
    meanObjs5 = numpy.squeeze(meanObjs5)
    
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    plt.figure(0)
    plt.title("X")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs1)
    plt.xlabel("alpha")
    plt.ylabel("objective")

    plt.figure(1)
    plt.title("X2")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs2)
    plt.xlabel("alpha")
    plt.ylabel("objective")

    plt.figure(2)
    plt.title("X3")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs3)
    plt.xlabel("alpha")
    plt.ylabel("objective")
    
    plt.figure(3)
    plt.title("X4")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs4)
    plt.xlabel("alpha")
    plt.ylabel("objective")

    plt.figure(4)
    plt.title("X5")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs5)
    plt.xlabel("alpha")
    plt.ylabel("objective")    
    

    plt.show()
    
print(meanObjs1)
print(meanObjs2)
print(meanObjs3)
print(meanObjs4)
print(meanObjs5)