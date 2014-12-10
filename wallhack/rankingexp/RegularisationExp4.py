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
Script to see if model selection is the same on a subset of rows or elements 
"""

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True
prefix = "Regularisation4"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 
X = DatasetUtils.getDataset(dataset)

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

u = 0.1
w = 1-u
k2 = 64
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k2, w, eps=eps, stochastic=True)
maxLocalAuc.alpha = 0.1
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.folds = 1
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.lmbdas = 2.0**-numpy.arange(0, 6)
maxLocalAuc.loss = "hinge"
maxLocalAuc.maxIterations = 500
maxLocalAuc.metric = "f1"
maxLocalAuc.normalise = True
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numProcesses = multiprocessing.cpu_count()
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.rate = "constant"
maxLocalAuc.recordStep = 10
maxLocalAuc.rho = 1.0
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.validationSize = 3
maxLocalAuc.validationUsers = 0

newM = X.shape[0]/4
modelSelectX, userInds = Sampling.sampleUsers(X, newM)

if saveResults: 
    meanObjs1, stdObjs1 = maxLocalAuc.modelSelect(X)
    meanObjs2, stdObjs2 = maxLocalAuc.modelSelect(trainX)
    meanObjs3, stdObjs3 = maxLocalAuc.modelSelect(modelSelectX)

    numpy.savez(outputFile, meanObjs1, meanObjs2, meanObjs3)
else: 
    data = numpy.load(outputFile)
    meanObjs1, meanObjs2, meanObjs3 = data["arr_0"], data["arr_1"], data["arr_2"]
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    plt.figure(0)
    plt.contourf(numpy.log2(maxLocalAuc.lmbdas), numpy.log2(maxLocalAuc.ks), meanObjs1)
    plt.xlabel("lambda")
    plt.ylabel("k")
    plt.colorbar()
    
    plt.figure(1)
    plt.contourf(numpy.log2(maxLocalAuc.lmbdas), numpy.log2(maxLocalAuc.ks), meanObjs2)
    plt.xlabel("lambda")
    plt.ylabel("k")
    plt.colorbar()
    
    plt.figure(2)    
    plt.contourf(numpy.log2(maxLocalAuc.lmbdas), numpy.log2(maxLocalAuc.ks), meanObjs3)
    plt.xlabel("lambda")
    plt.ylabel("k")
    plt.colorbar()
    
    plt.show()
    
print(meanObjs1)
print(meanObjs2)
print(meanObjs3)
