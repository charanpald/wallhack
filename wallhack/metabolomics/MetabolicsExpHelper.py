import os
import numpy
import sys
import logging
import datetime
import gc 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
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

class MetabolicsExpHelper(object):
    def __init__(self, dataDict, YCortisol, YTesto, YIgf1, ages):
        """
        Create a new object for run the metabolomics experiments
        """
        self.dataDict = dataDict
        self.runTreeRank = True 
        self.runTreeRankForest = True 
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

        self.rankBoost = RankBoost()
        self.rankBoostParams = {} 
        self.rankBoostParams["setIterations"] = numpy.array([10, 50, 100])
        self.rankBoostParams["setLearners"] = numpy.array([10, 20])
        
        self.rankSVM = RankSVM()
        self.rankSVM.setKernel("rbf")
        self.rankSVMParams = {} 
        self.rankSVMParams["setC"] = 2.0**numpy.arange(-5, -3, dtype=numpy.float)
        self.rankSVMParams["setGamma"] =  2.0**numpy.arange(-5, -3, dtype=numpy.float)

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
            logging.debug("Computing file " + fileName + " with learner " + str(learner))
            
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
                Y = numpy.array(hormoneIndicators[nonNaInds, i], numpy.int)    
                
                for dataName, dataFeatures in self.dataDict.items():
                    X = dataFeatures[nonNaInds, :]
                    X = numpy.c_[X, self.ages[nonNaInds]]
                    X = Standardiser().standardiseArray(X)

                    logging.debug("Shape of examples: " + str(X.shape))
                    logging.debug("Distribution of labels: " + str(numpy.bincount(Y)))

                    if self.runRankBoost: 
                        fileName = self.resultsDir + "RankBoost-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.rankBoost, self.rankBoostParams, fileName)
                        
                    if self.runRankSVM: 
                        fileName = self.resultsDir + "RankSVM-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.rankSVM, self.rankSVMParams, fileName)
                        
        logging.debug("All done, see you around!")
                        
    def run(self):
        logging.debug('module name:' + __name__) 
        logging.debug('parent process:' +  str(os.getppid()))
        logging.debug('process id:' +  str(os.getpid()))

        self.saveResults()
