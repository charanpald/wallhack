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
maxLocalAuc.maxIterations = 50
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
#maxLocalAuc.parallelSGD = True

os.system('taskset -p 0xffffffff %d' % os.getpid())

logging.debug("Starting training")
logging.debug(maxLocalAuc)

#modelSelectX = trainX[0:100, :]
#maxLocalAuc.learningRateSelect(trainX)
maxLocalAuc.modelSelect(trainX)
#ProfileUtils.profile('U, V, trainObjs, trainAucs, testObjs, testAucs, iterations, time = maxLocalAuc.learnModel(trainX, testX=testX, verbose=True)', globals(), locals())

U, V, trainMeasures, testMeasures, iterations, time = maxLocalAuc.learnModel(trainX, verbose=True)

p = 10

trainOrderedItems = MCEvaluator.recommendAtk(U, V, p)
testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, p, trainX)

r = SparseUtilsCython.computeR(U, V, maxLocalAuc.w, maxLocalAuc.numRecordAucSamples)
trainObjVec = maxLocalAuc.objectiveApprox(trainOmegaPtr, U, V, r, maxLocalAuc.gi, maxLocalAuc.gp, maxLocalAuc.gq, full=True)
testObjVec = maxLocalAuc.objectiveApprox(testOmegaPtr, U, V, r, maxLocalAuc.gi, maxLocalAuc.gp, maxLocalAuc.gq, allArray=allOmegaPtr, full=True)

itemCounts = numpy.array(X.sum(0)+1, numpy.int32)
beta = 0.5

for p in [1, 3, 5, 10]:
    trainPrecision = MCEvaluator.precisionAtK(trainOmegaPtr, trainOrderedItems, p)
    testPrecision = MCEvaluator.precisionAtK(testOmegaPtr, testOrderedItems, p)
    logging.debug("Train/test precision@" + str(p) + "=" + str(trainPrecision) + "/" + str(testPrecision)) 
    
for p in [1, 3, 5, 10]:
    trainRecall = MCEvaluator.stratifiedRecallAtK(trainOmegaPtr, trainOrderedItems, p, itemCounts, beta)
    testRecall = MCEvaluator.stratifiedRecallAtK(testOmegaPtr, testOrderedItems, p, itemCounts, beta)    
    logging.debug("Train/test stratified recall@" + str(p) + "=" + str(trainRecall) + "/" + str(testRecall))
    

plt.figure(0)
plt.plot(trainMeasures[:, 0], label="train")
plt.plot(testMeasures[:, 0], label="test") 
plt.xlabel("iteration")
plt.ylabel("objective")
plt.legend()


plt.figure(1)
plt.plot(trainMeasures[:, 1], label="train")
plt.plot(testMeasures[:, 1], label="test")
plt.xlabel("iteration")
plt.ylabel("local AUC")
plt.legend()

plt.figure(2)
plt.plot(testMeasures[:, 2])
plt.xlabel("iteration")
plt.ylabel("precision")
plt.legend()

plt.figure(3)
plt.plot(testMeasures[:, 3])
plt.xlabel("iteration")
plt.ylabel("mrr")
plt.legend()

#Look at distrubution of U and V 
Z = U.dot(V.T)

plt.figure(4)
Z2 = Z[X.toarray() == 0]
hist, edges = numpy.histogram(Z2.flatten(), bins=50, range=(-maxLocalAuc.lmbdaV, maxLocalAuc.lmbdaV) , normed=True)
xvals = (edges[0:-1]+edges[1:])/2
plt.plot(xvals, hist, label="zero")

trainVals = Z[trainX.nonzero()].flatten()
hist, e = numpy.histogram(trainVals, bins=edges, normed=True)
xvals = (edges[0:-1]+edges[1:])/2
plt.plot(xvals, hist, label="train")

testVals = Z[testX.nonzero()].flatten()
print(numpy.max(testVals))
hist, e = numpy.histogram(testVals, bins=edges, normed=True)
xvals = (edges[0:-1]+edges[1:])/2
plt.plot(xvals, hist, label="test")
plt.legend()


#Look at distribution of train and test objectives 
plt.figure(5)
hist, edges = numpy.histogram(trainObjVec, bins=50, normed=True)
xvals = (edges[0:-1]+edges[1:])/2
plt.plot(xvals, hist, label="train")

hist, e = numpy.histogram(testObjVec, bins=edges, normed=True)
xvals = (edges[0:-1]+edges[1:])/2
plt.plot(xvals, hist, label="test")
plt.legend()

plt.figure(6)
plt.scatter(trainObjVec, trainX.sum(1))

#See precisions 
f1s, orderedItems = MCEvaluator.f1AtK(testOmegaPtr, testOrderedItems, maxLocalAuc.recommendSize, verbose=True)
uniqp, inverse = numpy.unique(f1s, return_inverse=True)
print(uniqp, numpy.bincount(inverse))

numItems = trainX.sum(1)
print(numpy.corrcoef(numItems, f1s))
print(numpy.corrcoef(trainObjVec, f1s))
print(numpy.corrcoef(testObjVec, f1s))

#fprTrain, tprTrain = MCEvaluator.averageRocCurve(trainX, U, V)
#fprTest, tprTest = MCEvaluator.averageRocCurve(testX, U, V)
#
#plt.figure(7)
#plt.plot(fprTrain, tprTrain, label="train")
#plt.plot(fprTest, tprTest, label="test")
#plt.xlabel("mean false positive rate")
#plt.ylabel("mean true positive rate")
#plt.legend()
plt.show()