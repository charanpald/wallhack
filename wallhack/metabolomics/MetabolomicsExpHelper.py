import os
import numpy
import logging
import gc 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.FileLock import FileLock 
from apgl.util.Sampling import Sampling 
from apgl.util.Evaluator import Evaluator 
from sandbox.ranking.TreeRank import TreeRank
from sandbox.ranking.TreeRankForest import TreeRankForest
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from sandbox.data.Standardiser import Standardiser
from sandbox.ranking.leafrank.SVMLeafRank import SVMLeafRank
from sandbox.ranking.leafrank.DecisionTree import DecisionTree
from sandbox.ranking.RankSVM import RankSVM
from sandbox.ranking.RankBoost import RankBoost

class MetabolomicsExpHelper(object):
    def __init__(self, dataDict, YCortisol, YTesto, YIgf1, ages):
        """
        Create a new object for run the metabolomics experiments
        """
        self.dataDict = dataDict
        
        self.runCartTreeRank = False 
        self.runRbfSvmTreeRank = False 
        self.runCartTreeRankForest = False 
        self.runRbfSvmTreeRankForest = False 
        self.runRankBoost = False 
        self.runRankSVM = False 
        
        self.YCortisol = YCortisol 
        self.YTesto = YTesto 
        self.YIgf1 = YIgf1 
        self.ages = ages

        self.maxDepth = 10
        self.numTrees = 10
        self.sampleSize = 1.0
        self.sampleReplace = True
        
        self.outerFolds = 3
        self.innerFolds = 5
        self.resultsDir = PathDefaults.getOutputDir() + "metabolomics/"
     
        #CART TreeRank 
        leafRankFolds = 3 
        leafRankParamDict = {} 
        leafRankParamDict["setMaxDepth"] = numpy.arange(1, 8)
        leafRankLearner = DecisionTree(leafRankParamDict, leafRankFolds)  
     
        self.cartTreeRank = TreeRank(leafRankLearner)
        self.cartTreeRank.processes = 1
        self.cartTreeRankParams = {}
        self.cartTreeRankParams["setMaxDepth"] = numpy.array([2, 5, 10])     
     
        #RBF SVM TreeRank 
        leafRankFolds = 3 
        leafRankParamDict = {} 
        leafRankParamDict["setC"] = 2**numpy.arange(-5, 5, dtype=numpy.float)  
        leafRankParamDict["setGamma"] =  2.0**numpy.arange(-5, 0, dtype=numpy.float)
        leafRankLearner = SVMLeafRank(leafRankParamDict, leafRankFolds) 
        leafRankLearner.setKernel("rbf")
        
        self.rbfSvmTreeRank = TreeRank(leafRankLearner)
        self.rbfSvmTreeRank.processes = 1
        self.rbfSvmTreeRankParams = {}
        self.rbfSvmTreeRankParams["setMaxDepth"] = numpy.array([2, 5, 10])
        
        #CART TreeRankForest 
        leafRankFolds = 3 
        leafRankParamDict = {} 
        leafRankParamDict["setMaxDepth"] = numpy.arange(1, 8)
        leafRankLearner = DecisionTree(leafRankParamDict, leafRankFolds)  
     
        self.cartTreeRankForest = TreeRankForest(leafRankLearner)
        self.cartTreeRankForest.processes = 1
        self.cartTreeRankForest.setNumTrees(10)
        self.cartTreeRankForestParams = {}
        self.cartTreeRankForestParams["setMaxDepth"] = numpy.array([2, 5, 10])          
    
        #RBF SVM TreeRankForest 
        leafRankFolds = 3 
        leafRankParamDict = {} 
        leafRankParamDict["setC"] = 2**numpy.arange(-5, 5, dtype=numpy.float)  
        leafRankParamDict["setGamma"] =  2.0**numpy.arange(-5, 0, dtype=numpy.float)
        leafRankLearner = SVMLeafRank(leafRankParamDict, leafRankFolds) 
        leafRankLearner.setKernel("rbf")
     
        self.rbfSvmTreeRankForest = TreeRankForest(leafRankLearner)
        self.rbfSvmTreeRankForest.processes = 1
        self.rbfSvmTreeRankForest.setNumTrees(10)
        self.rbfSvmTreeRankForestParams = {}
        self.rbfSvmTreeRankForestParams["setMaxDepth"] = numpy.array([2, 5, 10])  
    
        #RankBoost 
        self.rankBoost = RankBoost()
        self.rankBoostParams = {} 
        self.rankBoostParams["setIterations"] = numpy.array([10, 50, 100])
        self.rankBoostParams["setLearners"] = numpy.array([5, 10, 20])
        
        #RankSVM
        self.rankSVM = RankSVM()
        self.rankSVM.setKernel("rbf")
        self.rankSVMParams = {} 
        self.rankSVMParams["setC"] = 2.0**numpy.arange(-5, 5, dtype=numpy.float)
        self.rankSVMParams["setGamma"] =  2.0**numpy.arange(-5, 0, dtype=numpy.float)

        #Store all the label vectors and their missing values
        self.hormoneDict = {"Cortisol": YCortisol, "Testosterone": YTesto, "IGF1": YIgf1}

    def saveResult(self, X, Y, learner, paramDict, fileName):
        """
        Save a single result to file, checking if the results have already been computed
        """
        filelock = FileLock(fileName)
        gc.collect()

        if not filelock.isLocked() and not filelock.fileExists(): 
            filelock.lock()
            try: 
                logging.debug("Computing file " + fileName)
                logging.debug("Shape of examples: " + str(X.shape) + ", number of +1: " + str(numpy.sum(Y==1)) + ", -1: " + str(numpy.sum(Y==-1)))
                
                idxFull = Sampling.crossValidation(self.outerFolds, X.shape[0])
                errors = numpy.zeros(self.outerFolds)
                
                for i, (trainInds, testInds) in enumerate(idxFull): 
                    logging.debug("Outer fold: " + str(i))
                    
                    trainX, trainY = X[trainInds, :], Y[trainInds]
                    testX, testY = X[testInds, :], Y[testInds]
                    idx = Sampling.crossValidation(self.innerFolds, trainX.shape[0])
                    bestLearner, cvGrid = learner.parallelModelSelect(trainX, trainY, idx, paramDict)
    
                    
                    bestLearner = learner.getBestLearner(cvGrid, paramDict, trainX, trainY, idx, best="max")
                    logging.debug("Best learner is " + str(bestLearner))
                    bestLearner.learnModel(trainX, trainY)
                    predY = bestLearner.predict(testX)
                    errors[i] = Evaluator.auc(predY, testY)
                
                logging.debug("Mean auc: " + str(numpy.mean(errors)))
                numpy.save(fileName, errors)
                logging.debug("Saved results as : " + fileName)
            finally: 
                filelock.unlock()
        else:
            logging.debug("File exists, or is locked: " + fileName)

    def saveResults(self):
        """
        Compute the results and save them for a particular hormone. Does so for all
        learners. 
        """
        metaUtils = MetabolomicsUtils()
        
        for hormoneName, hormoneConc in self.hormoneDict.items():
            nonNaInds = numpy.logical_not(numpy.isnan(hormoneConc))
            hormoneIndicators = metaUtils.createIndicatorLabel(hormoneConc, metaUtils.boundsDict[hormoneName])

            for i in range(hormoneIndicators.shape[1]):
                #Make labels -1/+1
                Y = numpy.array(hormoneIndicators[nonNaInds, i], numpy.int)*2-1    
                
                for dataName, dataFeatures in self.dataDict.items():
                    X = dataFeatures[nonNaInds, :]
                    X = numpy.c_[X, self.ages[nonNaInds]]
                    X = Standardiser().standardiseArray(X)

                    

                    if self.runCartTreeRank: 
                        fileName = self.resultsDir + "CartTreeRank-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.cartTreeRank, self.cartTreeRankParams, fileName) 
                        
                    if self.runRbfSvmTreeRank: 
                        fileName = self.resultsDir + "RbfSvmTreeRank-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.rbfSvmTreeRank, self.rbfSvmTreeRankParams, fileName)    

                    if self.runCartTreeRankForest: 
                        fileName = self.resultsDir + "CartTreeRankForest-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.cartTreeRankForest, self.cartTreeRankForestParams, fileName) 
                        
                    if self.runRbfSvmTreeRankForest: 
                        fileName = self.resultsDir + "RbfSvmTreeRankForest-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.rbfSvmTreeRankForest, self.rbfSvmTreeRankForestParams, fileName) 

                    if self.runRankBoost: 
                        fileName = self.resultsDir + "RankBoost-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.rankBoost, self.rankBoostParams, fileName)
                        
                    if self.runRankSVM: 
                        fileName = self.resultsDir + "RankSVM-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.rankSVM, self.rankSVMParams, fileName)
                        
        logging.debug("All done. See you around!")
                        
    def run(self):
        logging.debug('module name:' + __name__) 
        logging.debug('parent process:' +  str(os.getppid()))
        logging.debug('process id:' +  str(os.getpid()))

        self.saveResults()
