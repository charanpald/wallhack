import os
import numpy
import logging
import gc 
try: 
    from sklearn.cross_val import StratifiedKFold
except: 
    #Version 0.12 fix 
    from sklearn.cross_validation import StratifiedKFold

from apgl.util.PathDefaults import PathDefaults
from apgl.util.FileLock import FileLock 
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
    def __init__(self, dataDict, YCortisol, YTesto, YIgf1, ages, numProcesses=1, runCortisol=True, runTestosterone=True, runIGF1=True):
        """
        Create a new object for run the metabolomics experiments
        """
        self.dataDict = dataDict
        
        self.runCartTreeRank = False 
        self.runRbfSvmTreeRank = False 
        self.runL1SvmTreeRank = False
        self.runCartTreeRankForest = False 
        self.runRbfSvmTreeRankForest = False 
        self.runL1SvmTreeRankForest = False
        self.runRankBoost = False 
        self.runRankSVM = False 
        
        self.runCortisol = runCortisol 
        self.runTestosterone = runTestosterone 
        self.runIGF1 = runIGF1
        
        self.YCortisol = YCortisol 
        self.YTesto = YTesto 
        self.YIgf1 = YIgf1 
        self.ages = ages

        self.outerFolds = 3
        self.innerFolds = 5
        self.leafRankFolds = 3
        self.resultsDir = PathDefaults.getOutputDir() + "metabolomics/"
        self.numProcesses = numProcesses

        #General params 
        Cs = 2.0**numpy.arange(-5, 7, 2, dtype=numpy.float)   
        gammas = 2.0**numpy.arange(-5, 3, 2, dtype=numpy.float)
        depths = numpy.array([2, 4, 8]) 
        numTrees = 20
        sampleSize = 1.0
        maxDepth = 10
        featureSize = 0.5 

        #CART TreeRank 
        leafRankParamDict = {} 
        leafRankParamDict["setMaxDepth"] = depths
        leafRankLearner = DecisionTree(leafRankParamDict, self.leafRankFolds)  
     
        self.cartTreeRank = TreeRank(leafRankLearner, numProcesses=numProcesses)
        self.cartTreeRankParams = {}
        self.cartTreeRankParams["setMaxDepth"] = depths
     
        #RBF SVM TreeRank 
        leafRankParamDict = {} 
        leafRankParamDict["setC"] = Cs  
        leafRankParamDict["setGamma"] =  gammas
        leafRankLearner = SVMLeafRank(leafRankParamDict, self.leafRankFolds) 
        leafRankLearner.setKernel("rbf")
        leafRankLearner.processes = 1
        
        self.rbfSvmTreeRank = TreeRank(leafRankLearner, numProcesses=numProcesses)
        self.rbfSvmTreeRankParams = {}
        self.rbfSvmTreeRankParams["setMaxDepth"] = depths
        
        #Linear L1 SVM TreeRank 
        leafRankParamDict = {} 
        leafRankParamDict["setC"] = Cs 
        leafRankLearner = SVMLeafRank(leafRankParamDict, self.leafRankFolds) 
        leafRankLearner.setKernel("linear")
        leafRankLearner.setPenalty("l1")
        leafRankLearner.processes = 1
        
        self.l1SvmTreeRank = TreeRank(leafRankLearner, numProcesses=numProcesses)
        self.l1SvmTreeRankParams = {}
        self.l1SvmTreeRankParams["setMaxDepth"] = depths       
        
        #CART TreeRankForest 
        leafRankParamDict = {} 
        leafRankParamDict["setMaxDepth"] = depths 
        leafRankLearner = DecisionTree(leafRankParamDict, self.leafRankFolds)  
        leafRankLearner.processes = 1
     
        self.cartTreeRankForest = TreeRankForest(leafRankLearner, numProcesses=numProcesses)
        self.cartTreeRankForest.setNumTrees(numTrees)
        self.cartTreeRankForest.setSampleSize(sampleSize)
        self.cartTreeRankForest.setFeatureSize(featureSize)
        self.cartTreeRankForestParams = {}
        self.cartTreeRankForestParams["setMaxDepth"] = numpy.array([maxDepth])   
        self.cartTreeRankForestParams["setSampleSize"] = numpy.array([0.5, 0.75, 1.0])
        self.cartTreeRankForestParams["setFeatureSize"] = numpy.array([0.5, 0.75, 1.0])
    
        #RBF SVM TreeRankForest 
        leafRankParamDict = {} 
        leafRankParamDict["setC"] = Cs  
        leafRankParamDict["setGamma"] =  gammas
        leafRankLearner = SVMLeafRank(leafRankParamDict, self.leafRankFolds) 
        leafRankLearner.setKernel("rbf")
        leafRankLearner.processes = 1
     
        self.rbfSvmTreeRankForest = TreeRankForest(leafRankLearner, numProcesses=numProcesses)
        self.rbfSvmTreeRankForest.setNumTrees(numTrees)
        self.rbfSvmTreeRankForest.setSampleSize(sampleSize)
        self.rbfSvmTreeRankForest.setFeatureSize(featureSize)
        self.rbfSvmTreeRankForestParams = {}
        self.rbfSvmTreeRankForestParams["setMaxDepth"] = numpy.array([maxDepth]) 
        self.rbfSvmTreeRankForestParams["setSampleSize"] = numpy.array([0.5, 0.75, 1.0])
        self.rbfSvmTreeRankForestParams["setFeatureSize"] = numpy.array([0.5, 0.75, 1.0])
    
        #L1 SVM TreeRankForest 
        leafRankParamDict = {} 
        leafRankParamDict["setC"] = Cs 
        leafRankLearner = SVMLeafRank(leafRankParamDict, self.leafRankFolds) 
        leafRankLearner.setKernel("linear")
        leafRankLearner.setPenalty("l1")  
        leafRankLearner.processes = 1
        
        self.l1SvmTreeRankForest = TreeRankForest(leafRankLearner, numProcesses=numProcesses)
        self.l1SvmTreeRankForest.setNumTrees(numTrees)
        self.l1SvmTreeRankForest.setSampleSize(sampleSize)
        self.l1SvmTreeRankForest.setFeatureSize(featureSize)
        self.l1SvmTreeRankForestParams = {}
        self.l1SvmTreeRankForestParams["setMaxDepth"] = numpy.array([maxDepth]) 
        self.l1SvmTreeRankForestParams["setSampleSize"] = numpy.array([0.5, 0.75, 1.0])
        self.l1SvmTreeRankForestParams["setFeatureSize"] = numpy.array([0.5, 0.75, 1.0])
    
        #RankBoost 
        self.rankBoost = RankBoost(numProcesses=numProcesses)
        self.rankBoostParams = {} 
        self.rankBoostParams["setIterations"] = numpy.array([10, 50, 100])
        self.rankBoostParams["setLearners"] = numpy.array([5, 10, 20])
        
        #RankSVM
        self.rankSVM = RankSVM(numProcesses=numProcesses)
        self.rankSVM.setKernel("rbf")
        self.rankSVMParams = {} 
        self.rankSVMParams["setC"] = 2.0**numpy.arange(0, 3, dtype=numpy.float)
        self.rankSVMParams["setGamma"] =  2.0**numpy.arange(-3, 0, dtype=numpy.float)

        #Store all the label vectors and their missing values
        self.hormoneDict = {}
        if self.runCortisol: 
            self.hormoneDict["Cortisol"] = YCortisol
        if self.runTestosterone: 
            self.hormoneDict["Testosterone"] = YTesto
        if self.runIGF1: 
            self.hormoneDict["IGF1"] = YIgf1
        

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
                
                #idxFull = Sampling.crossValidation(self.outerFolds, X.shape[0])
                idxFull = StratifiedKFold(Y, self.outerFolds)
                errors = numpy.zeros(self.outerFolds)
                
                for i, (trainInds, testInds) in enumerate(idxFull): 
                    logging.debug("Outer fold: " + str(i))
                    
                    trainX, trainY = X[trainInds, :], Y[trainInds]
                    testX, testY = X[testInds, :], Y[testInds]
                    #idx = Sampling.crossValidation(self.innerFolds, trainX.shape[0])
                    idx = StratifiedKFold(trainY, self.innerFolds)
                    logging.debug("Initial learner is " + str(learner))
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

    def saveWeightVectorResults(self, X, Y, learner, paramDict, fileName): 
        """
        Save the results of the variable importance 
        """
        filelock = FileLock(fileName)
        gc.collect()

        if not filelock.isLocked() and not filelock.fileExists(): 
            filelock.lock()
            try: 
                logging.debug("Computing weights file " + fileName)
                logging.debug("Shape of examples: " + str(X.shape) + ", number of +1: " + str(numpy.sum(Y==1)) + ", -1: " + str(numpy.sum(Y==-1)))
                                
                logging.debug("Initial learner is " + str(learner))
                idx = StratifiedKFold(Y, self.innerFolds)
                bestLearner, cvGrid = learner.parallelModelSelect(X, Y, idx, paramDict)

                bestLearner = learner.getBestLearner(cvGrid, paramDict, X, Y, idx, best="max")
                logging.debug("Best learner is " + str(bestLearner))
                
                bestLearner.learnModel(X, Y)
                weightVector = bestLearner.variableImportance(X, Y)   
                numpy.save(fileName, weightVector)
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
        
        logging.debug("Running on hormones: " + str(self.hormoneDict.keys()))
        
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

                    if self.runL1SvmTreeRank: 
                        fileName = self.resultsDir + "L1SvmTreeRank-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.l1SvmTreeRank, self.l1SvmTreeRankParams, fileName)   
                        
                        #For this SVM save the weight vector 
                        weightsFileName = self.resultsDir + "WeightsL1SvmTreeRank-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.l1SvmTreeRank, self.l1SvmTreeRankParams, weightsFileName)    

                    if self.runCartTreeRankForest: 
                        fileName = self.resultsDir + "CartTreeRankForest-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.cartTreeRankForest, self.cartTreeRankForestParams, fileName) 
                        
                    if self.runRbfSvmTreeRankForest: 
                        fileName = self.resultsDir + "RbfSvmTreeRankForest-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.rbfSvmTreeRankForest, self.rbfSvmTreeRankForestParams, fileName) 
                        
                    if self.runL1SvmTreeRankForest: 
                        fileName = self.resultsDir + "L1SvmTreeRankForest-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.l1SvmTreeRankForest, self.l1SvmTreeRankForestParams, fileName) 
                        
                        #For this SVM save the weight vector 
                        weightsFileName = self.resultsDir + "WeightsL1SvmTreeRankForest-" + hormoneName + "-" + str(i) + "-" + dataName + ".npy"
                        self.saveResult(X, Y, self.l1SvmTreeRankForest, self.l1SvmTreeRankForestParams, weightsFileName)    

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
