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
    dataset = "epinions"

saveResults = True
prefix = "LearningRate"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 
X = DatasetUtils.getDataset(dataset, nnz=200000)

u = 0.1
w = 1-u
k2 = 64
eps = 10**-6
maxLocalAuc = MaxLocalAUC(k2, w, eps=eps, stochastic=True)
maxLocalAuc.alpha = 0.1
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 8, 1)
maxLocalAuc.folds = 2
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.lmbdas = 2.0**-numpy.arange(0, 8)
maxLocalAuc.lmbdaU = 0.05
maxLocalAuc.lmbdaV = 0.05
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

newM = X.nnz/10
modelSelectX, userInds = Sampling.sampleUsers2(X, newM, prune=True)

print(X.shape, X.nnz)
print(modelSelectX.shape, modelSelectX.nnz)

if saveResults: 
    meanObjs1, stdObjs1 = maxLocalAuc.learningRateSelect(X)
    meanObjs2, stdObjs2 = maxLocalAuc.learningRateSelect(modelSelectX)

    numpy.savez(outputFile, meanObjs1, meanObjs2)
else: 
    data = numpy.load(outputFile)
    meanObjs1, meanObjs2 = data["arr_0"], data["arr_1"]
    
    meanObjs1 = numpy.squeeze(meanObjs1)    
    meanObjs2 = numpy.squeeze(meanObjs2) 
    
    
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    plt.figure(0)
    plt.title("X")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs1)
    plt.xlabel("alpha")
    plt.ylabel("lambda")
    plt.colorbar()
    
    plt.figure(1)
    plt.title("modelSelectX")
    plt.plot(numpy.log2(maxLocalAuc.alphas), meanObjs2)
    plt.xlabel("alpha")
    plt.ylabel("lambda")
    plt.colorbar()
    
    plt.show()
    
print(meanObjs1)
print(meanObjs2)
