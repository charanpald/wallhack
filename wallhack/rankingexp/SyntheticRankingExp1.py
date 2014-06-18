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
    dataset = "movielens"

saveResults = True

if dataset == "synthetic": 
    X, U, V = DatasetUtils.syntheticDataset1()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp1SyntheticResults.npz" 
elif dataset == "synthetic2": 
    X, U, V = DatasetUtils.syntheticDataset1(u=0.5)
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp1Synthetic2Results.npz" 
elif dataset == "synthetic3": 
    X, U, V = DatasetUtils.syntheticDataset1(u=0.2, sd=0.2)
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp1Synthetic3Results.npz" 
elif dataset == "movielens": 
    X = DatasetUtils.movieLens()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp1MovieLensResults.npz" 
elif dataset == "flixster": 
    X = DatasetUtils.flixster()
    outputFile = PathDefaults.getOutputDir() + "ranking/Exp1FlixsterResults.npz" 
    X = Sampling.sampleUsers(X, 1000)
else: 
    raise ValueError("Unknown dataset: " + dataset)

m,n = X.shape
u = 0.1 
w = 1-u

print(numpy.histogram(X.sum(0)))

testSize = 5
trainTestXs = Sampling.shuffleSplitRows(X, 1, testSize)
trainX, testX = trainTestXs[0]

trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
allOmegaPtr = SparseUtils.getOmegaListPtr(X)
numRecordAucSamples = 200

logging.debug("Number of non-zero elements: " + str((trainX.nnz, testX.nnz)))
if dataset == "synthetic": 
    logging.debug("Train local AUC:" + str(MCEvaluator.localAUCApprox(trainOmegaPtr, U, V, w, numRecordAucSamples, allArray=allOmegaPtr)))
    logging.debug("Test local AUC:" + str(MCEvaluator.localAUCApprox(testOmegaPtr, U, V, w, numRecordAucSamples, allArray=allOmegaPtr)))


#w = 1.0
k2 = 64
u2 = 0.1
w2 = 1-u2
eps = 10**-8
lmbda = 1.0
maxLocalAuc = MaxLocalAUC(k2, w2, eps=eps, lmbda=lmbda, stochastic=True)
maxLocalAuc.maxIterations = 200
maxLocalAuc.numRowSamples = 30
maxLocalAuc.numAucSamples = 5
maxLocalAuc.numRecordAucSamples = 100
maxLocalAuc.recordStep = 5
maxLocalAuc.initialAlg = "rand"
maxLocalAuc.rate = "optimal"
maxLocalAuc.alpha = 0
maxLocalAuc.t0 = 0.5
maxLocalAuc.folds = 2
maxLocalAuc.rho = 0.1
maxLocalAuc.ks = numpy.array([k2])
maxLocalAuc.validationSize = 3
maxLocalAuc.lmbdas = numpy.linspace(0.5, 2.0, 7)
maxLocalAuc.normalise = True
#maxLocalAuc.numProcesses = 1
maxLocalAuc.alphas = 2.0**-numpy.arange(0, 5, 1)
maxLocalAuc.t0s = 2.0**-numpy.arange(7, 12, 1)
maxLocalAuc.metric = "precision"
maxLocalAuc.sampling = "uniform"
#maxLocalAuc.numProcesses = 1

os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
logging.debug(maxLocalAuc)

#modelSelectX = trainX[0:100, :]
#maxLocalAuc.learningRateSelect(trainX)
#maxLocalAuc.modelSelect(trainX)
#ProfileUtils.profile('U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, time = maxLocalAuc.learnModel(trainX, testX=testX, verbose=True)', globals(), locals())

U, V, trainObjs, trainAucs, testObjs, testAucs, precisions, iterations, time = maxLocalAuc.learnModel(trainX, verbose=True)

p = 10

trainOrderedItems = MCEvaluator.recommendAtk(U, V, p)
testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, p, trainX)

r = SparseUtilsCython.computeR(U, V, maxLocalAuc.w, maxLocalAuc.numRecordAucSamples)
trainObjVec = maxLocalAuc.objectiveApprox(trainOmegaPtr, U, V, r, full=True)
testObjVec = maxLocalAuc.objectiveApprox(testOmegaPtr, U, V, r, allArray=allOmegaPtr, full=True)

print(trainObjVec)
print(testObjVec)

for p in [1, 3, 5, 10]: 
    logging.debug("Train precision@" + str(p) + "=" + str(MCEvaluator.precisionAtK(trainOmegaPtr, trainOrderedItems, p))) 
    logging.debug("Test precision@" + str(p) + "=" + str(MCEvaluator.precisionAtK(testOmegaPtr, testOrderedItems, p))) 

plt.figure(0)
plt.plot(trainObjs, label="train")
plt.plot(testObjs, label="test") 
plt.xlabel("iteration")
plt.ylabel("objective")
plt.legend()


plt.figure(1)
plt.plot(trainAucs, label="train")
plt.plot(testAucs, label="test")
plt.xlabel("iteration")
plt.ylabel("local AUC")
plt.legend()

plt.figure(2)
plt.plot(precisions)
plt.xlabel("iteration")
plt.ylabel("precision")
plt.legend()


#Look at distrubution of U and V 
Z = U.dot(V.T)

plt.figure(3)
Z2 = Z[X.toarray() == 0]
hist, edges = numpy.histogram(Z2.flatten(), bins=50, normed=True)
xvals = (edges[0:-1]+edges[1:])/2
plt.plot(xvals, hist, label="zero")

trainVals = Z[trainX.nonzero()].flatten()
hist, e = numpy.histogram(trainVals, bins=edges, normed=True)
xvals = (edges[0:-1]+edges[1:])/2
plt.plot(xvals, hist, label="train")

testVals = Z[testX.nonzero()].flatten()
hist, e = numpy.histogram(testVals, bins=edges, normed=True)
xvals = (edges[0:-1]+edges[1:])/2
plt.plot(xvals, hist, label="test")
plt.legend()


#Look at distribution of train and test objectives 
plt.figure(4)
hist, edges = numpy.histogram(trainObjVec, bins=50, normed=True)
xvals = (edges[0:-1]+edges[1:])/2
plt.plot(xvals, hist, label="train")

hist, e = numpy.histogram(testObjVec, bins=edges, normed=True)
xvals = (edges[0:-1]+edges[1:])/2
plt.plot(xvals, hist, label="test")
plt.legend()

plt.figure(5)
plt.scatter(trainObjVec, trainX.sum(1))

#fprTrain, tprTrain = MCEvaluator.averageRocCurve(trainX, U, V)
#fprTest, tprTest = MCEvaluator.averageRocCurve(testX, U, V)
#
#plt.figure(4)
#plt.plot(fprTrain, tprTrain, label="train")
#plt.plot(fprTest, tprTest, label="test")
#plt.xlabel("mean false positive rate")
#plt.ylabel("mean true positive rate")
#plt.legend()
plt.show()

