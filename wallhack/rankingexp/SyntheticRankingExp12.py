import numpy
import logging
import sys
import os
import sppy
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.util.SparseUtils import SparseUtils
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from sandbox.util.ProfileUtils import ProfileUtils
from sandbox.util.MCEvaluator import MCEvaluator
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Sampling import Sampling
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython
from sandbox.util.SparseUtilsCython import SparseUtilsCython
from wallhack.rankingexp.DatasetUtils import DatasetUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=4, suppress=True, linewidth=150)
numpy.seterr(all="raise")

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else: 
    dataset = "synthetic"

saveResults = True

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp1Synthetic1Results.npz" 
elif dataset == "synthetic2": 
    X = DatasetUtils.syntheticDataset2()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp1Synthetic2Results.npz" 
elif dataset == "movielens": 
    X = DatasetUtils.movieLens(minNnzRows=20)
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp1MovieLensResults.npz" 
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp1FlixsterResults.npz" 
    X = Sampling.sampleUsers(X, 1000)
else: 
    raise ValueError("Unknown dataset: " + dataset)

m, n = X.shape
u = 0.1 
w = 1-u

logging.debug(numpy.histogram(X.sum(1)))
logging.debug(numpy.histogram(X.sum(0)))

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
allOmegaPtr = SparseUtils.getOmegaListPtr(X)
numRecordAucSamples = 200

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))

#w = 1.0
k2 = 64
u2 = 5/float(n)
w2 = 1-u2
eps = 10**-8
lmbda = 10**-3
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbdaV=lmbda, stochastic=True)
maxLocalAuc.maxIterations = 100
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 10
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.recordStep = 10
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 4.0
maxLocalAuc.t0 = 0.20
maxLocalAuc.folds = 2
maxLocalAuc.rho = 0.0
maxLocalAuc.lmbdaU = 0.0
maxLocalAuc.lmbdaV = 10.0
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.validationSize = 3
maxLocalAuc.z = 10
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.normalise = True
#maxLocalAuc.numProcesses = 1
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.metric = "mrr"
maxLocalAuc.sampling = "uniform"
maxLocalAuc.itemExpP = 0.0
maxLocalAuc.itemExpQ = 0.0
maxLocalAuc.itemFactors = False
maxLocalAuc.parallelSGD = False

os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
logging.debug(maxLocalAuc)

numpy.random.seed(21)
U, V = maxLocalAuc.learnModel(trainX)

