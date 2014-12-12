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
Script to see if model selection is the same on a subset of rows or elements. We 
use bounds on the rows of U and V. 
"""

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True
prefix = "Regularisation5"
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
maxLocalAuc.lmbdas = 2.0**-numpy.arange(0, 8)
maxLocalAuc.loss = "hinge"
maxLocalAuc.maxIterations = 500
maxLocalAuc.maxNorms = 2.0**numpy.arange(-2, 2, 0.5)
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
    meanObjs1, stdObjs1 = maxLocalAuc.modelSelect2(X)
    meanObjs2, stdObjs2 = maxLocalAuc.modelSelect2(trainX)
    meanObjs3, stdObjs3 = maxLocalAuc.modelSelect2(modelSelectX)

    numpy.savez(outputFile, meanObjs1, meanObjs2, meanObjs3)
else: 
    data = numpy.load(outputFile)
    meanObjs1, meanObjs2, meanObjs3 = data["arr_0"], data["arr_1"], data["arr_2"]
    
    meanObjs1 = numpy.squeeze(meanObjs1)    
    meanObjs2 = numpy.squeeze(meanObjs2) 
    meanObjs3 = numpy.squeeze(meanObjs3) 
    
    
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    plt.figure(0)
    plt.title("X")
    plt.contourf(numpy.log2(maxLocalAuc.alphas), numpy.log2(maxLocalAuc.maxNorms), meanObjs1)
    plt.xlabel("alpha")
    plt.ylabel("maxNorm")
    plt.colorbar()
    
    plt.figure(1)
    plt.title("trainX")
    plt.contourf(numpy.log2(maxLocalAuc.alphas), numpy.log2(maxLocalAuc.maxNorms), meanObjs2)
    plt.xlabel("alpha")
    plt.ylabel("maxNorm")
    plt.colorbar()
    
    plt.figure(2) 
    plt.title("modelSelectX")
    plt.contourf(numpy.log2(maxLocalAuc.alphas), numpy.log2(maxLocalAuc.maxNorms), meanObjs3)
    plt.xlabel("alpha")
    plt.ylabel("maxNorm")
    plt.colorbar()
    
    plt.show()
    
print(meanObjs1)
print(meanObjs2)
print(meanObjs3)
