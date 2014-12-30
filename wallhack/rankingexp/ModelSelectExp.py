
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
Script to see if the model selection is the same on a subset of rows or elements. 
"""

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "flixster"

saveResults = True
prefix = "ModelSelect"
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
maxLocalAuc.lmbdas = 2.0**-numpy.arange(2, 5)
maxLocalAuc.lmbdaU = 0.25
maxLocalAuc.lmbdaV = 0.25
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
    
    meanF1s1, stdF1s1 = maxLocalAuc.modelSelect(X)
    meanF1s2, stdF1s2 = maxLocalAuc.modelSelect(X2)
    meanF1s3, stdF1s3 = maxLocalAuc.modelSelect(X3)
    meanF1s4, stdF1s4 = maxLocalAuc.modelSelect(X4)
    meanF1s5, stdF1s5 = maxLocalAuc.modelSelect(X5)

    numpy.savez(outputFile, meanF1s1, meanF1s2, meanF1s3, meanF1s4, meanF1s5)
else: 
    data = numpy.load(outputFile)
    meanF1s1, meanF1s2, meanF1s3, meanF1s4, meanF1s5 = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"]
    
    meanF1s1 = numpy.squeeze(meanF1s1)    
    meanF1s2 = numpy.squeeze(meanF1s2) 
    meanF1s3 = numpy.squeeze(meanF1s3)
    meanF1s4 = numpy.squeeze(meanF1s4)
    meanF1s5 = numpy.squeeze(meanF1s5)
    
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    plt.figure(0)
    plt.title("X")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanF1s1)
    plt.xlabel("alpha")
    plt.ylabel("F1")

    plt.figure(1)
    plt.title("X2")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanF1s2)
    plt.xlabel("alpha")
    plt.ylabel("F1")

    plt.figure(2)
    plt.title("X3")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanF1s3)
    plt.xlabel("alpha")
    plt.ylabel("F1")
    
    plt.figure(3)
    plt.title("X4")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanF1s4)
    plt.xlabel("alpha")
    plt.ylabel("F1")

    plt.figure(4)
    plt.title("X5")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanF1s5)
    plt.xlabel("alpha")
    plt.ylabel("F1")    
    

    plt.show()
    
print(meanF1s1)
print(meanF1s2)
print(meanF1s3)
print(meanF1s4)
print(meanF1s5)