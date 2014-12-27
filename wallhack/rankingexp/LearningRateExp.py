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

u = 0.1
w = 1-u
k2 = 64
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k2, w, eps=eps, stochastic=True)
maxLocalAuc.alpha = 0.1
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 8, 1)
maxLocalAuc.folds = 1
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.lmbdas = 2.0**-numpy.arange(0, 8)
maxLocalAuc.lmbdaU = 0.5
maxLocalAuc.lmbdaV = 0.5
maxLocalAuc.loss = "hinge"
maxLocalAuc.maxIterations = 500
maxLocalAuc.maxNormU = 100
maxLocalAuc.maxNormV = 100
maxLocalAuc.metric = "f1"
maxLocalAuc.normalise = True
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numProcesses = multiprocessing.cpu_count()
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.numRowSamples = 15
maxLocalAuc.rate = "constant"
maxLocalAuc.recordStep = 10
maxLocalAuc.rho = 1.0
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationSize = 3
maxLocalAuc.validationUsers = 0

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
    
    meanObjs1, stdObjs1 = maxLocalAuc.learningRateSelect(X)
    meanObjs2, stdObjs2 = maxLocalAuc.learningRateSelect(X2)
    meanObjs3, stdObjs3 = maxLocalAuc.learningRateSelect(X3)
    meanObjs4, stdObjs4 = maxLocalAuc.learningRateSelect(X4)
    meanObjs5, stdObjs5 = maxLocalAuc.learningRateSelect(X5)

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
    plt.title("modelSelectX")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs2)
    plt.xlabel("alpha")
    plt.ylabel("objective")

    plt.figure(2)
    plt.title("modelSelectX")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs3)
    plt.xlabel("alpha")
    plt.ylabel("objective")
    
    plt.figure(3)
    plt.title("modelSelectX")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs4)
    plt.xlabel("alpha")
    plt.ylabel("objective")

    plt.figure(4)
    plt.title("modelSelectX")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs5)
    plt.xlabel("alpha")
    plt.ylabel("objective")    
    

    plt.show()
    
print(meanObjs1)
print(meanObjs2)
print(meanObjs3)
print(meanObjs4)
print(meanObjs5)