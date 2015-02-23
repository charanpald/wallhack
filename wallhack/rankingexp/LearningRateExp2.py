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
Script to see if there is an advantage of having independent learning rates alphaU and alphaV 
"""

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "movielens"

saveResults = True
prefix = "LearningRate2"
outputFile = PathDefaults.getOutputDir() + "ranking/" + prefix + dataset.title() + "Results.npz" 
X = DatasetUtils.getDataset(dataset)
m, n = X.shape

k2 = 64
u2 = 5/float(n)
w2 = 1-u2
eps = 10**-8
lmbda = 0.01
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaU=0.1, lmbdaV=0.1, stochastic=True)
maxLocalAuc.alpha = 0.5
maxLocalAuc.alphas = 2.0**-numpy.arange(2, 9, 2)
maxLocalAuc.beta = 2
maxLocalAuc.bound = False
maxLocalAuc.delta = 0.1
maxLocalAuc.eta = 20
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
maxLocalAuc.numRowSamples = 15
maxLocalAuc.rate = "optimal"
maxLocalAuc.recordStep = 10
maxLocalAuc.reg = False
maxLocalAuc.rho = 1.0
maxLocalAuc.startAverage = 100
maxLocalAuc.t0 = 1.0
maxLocalAuc.t0s = 2.0**-numpy.arange(1, 12, 2)
maxLocalAuc.validationSize = 5
maxLocalAuc.validationUsers = 0.0

if saveResults: 
    X = DatasetUtils.getDataset(dataset, nnz=100000)
    print(X.shape, X.nnz)
    print(maxLocalAuc)

    maxLocalAuc.lmbdaU = 0.25
    maxLocalAuc.lmbdaV = 0.25
    meanObjs1, paramDict = maxLocalAuc.learningRateSelect(X)

    maxLocalAuc.lmbdaU = 0.03125
    maxLocalAuc.lmbdaV = 0.25
    meanObjs2, paramDict = maxLocalAuc.learningRateSelect(X)

    maxLocalAuc.lmbdaU = 0.25
    maxLocalAuc.lmbdaV = 0.03125
    meanObjs3, paramDict = maxLocalAuc.learningRateSelect(X)
    
    maxLocalAuc.lmbdaU = 0.03125
    maxLocalAuc.lmbdaV = 0.03125
    meanObjs4, paramDict = maxLocalAuc.learningRateSelect(X)

    numpy.savez(outputFile, meanObjs1, meanObjs2, meanObjs3, meanObjs4)
else: 
    data = numpy.load(outputFile)
    meanObjs1, meanObjs2, meanObjs3, meanObjs4 = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
    
    print(meanObjs1)
    print(meanObjs2)
    print(meanObjs3)
    print(meanObjs4)
    meanObjs1[4,4] = meanObjs1[3,3]   
    meanObjs2[4,4] = meanObjs2[3,3] 
    meanObjs3[4,4] = meanObjs3[3,3] 
    meanObjs4[4,4] = meanObjs4[3,3] 
    
    import matplotlib 
    matplotlib.use("GTK3Agg")
    import matplotlib.pyplot as plt 
    plt.figure(0)
    plt.title("X")
    plt.contourf(numpy.log2(maxLocalAuc.alphas), numpy.log2(maxLocalAuc.alphas), meanObjs1)
    plt.xlabel("alphaU")
    plt.ylabel("alphaV")
    plt.colorbar()

    plt.figure(1)
    plt.title("X")
    plt.contourf(numpy.log2(maxLocalAuc.alphas), numpy.log2(maxLocalAuc.alphas), meanObjs2)
    plt.xlabel("alphaU")
    plt.ylabel("alphaV")
    plt.colorbar()

    plt.figure(2)
    plt.title("X")
    plt.contourf(numpy.log2(maxLocalAuc.alphas), numpy.log2(maxLocalAuc.alphas), meanObjs3)
    plt.xlabel("alphaU")
    plt.ylabel("alphaV")
    plt.colorbar()
    
    plt.figure(3)
    plt.title("X")
    plt.contourf(numpy.log2(maxLocalAuc.alphas), numpy.log2(maxLocalAuc.alphas), meanObjs4)
    plt.xlabel("alphaU")
    plt.ylabel("alphaV")
    plt.colorbar() 
    
    plt.show()
    
print(meanObjs1)
print(meanObjs2)
print(meanObjs3)
print(meanObjs4)