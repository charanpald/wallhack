
"""
Some common functions used for the recommendation experiments 
"""
import os
import gc 
import logging
import numpy
import argparse
import time 
from copy import copy
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.recommendation.SoftImpute import SoftImpute
from sandbox.util.SparseUtils import SparseUtils 
from sandbox.util.Sampling import Sampling 
from sandbox.util.FileLock import FileLock 

class RankingExpHelper(object):
    defaultAlgoArgs = argparse.Namespace()
    defaultAlgoArgs.runMaxLocalAuc = False
    defaultAlgoArgs.runSoftImpute = False
    defaultAlgoArgs.rhos = numpy.linspace(0.5, 0.0, 6)     
    defaultAlgoArgs.folds = 4
    defaultAlgoArgs.u = 0.2
    defaultAlgoArgs.eps = 0.01
    defaultAlgoArgs.sigma = 0.2
    defaultAlgoArgs.numRowSamples = 50
    defaultAlgoArgs.numColSamples = 50
    defaultAlgoArgs.numAucSamples = 50
    defaultAlgoArgs.trainSplit = 2.0/3
    defaultAlgoArgs.modelSelect = False
    defaultAlgoArgs.postProcess = False 
    defaultAlgoArgs.trainError = False 
    defaultAlgoArgs.verbose = False
    
    def __init__(self, cmdLine=None, defaultAlgoArgs = None, dirName=""):
        """ priority for default args
         - best priority: command-line value
         - middle priority: set-by-function value
         - lower priority: class value
        """
        # Parameters to choose which methods to run
        # Obtained merging default parameters from the class with those from the user
        self.algoArgs = RankingExpHelper.newAlgoParams(defaultAlgoArgs)
        
        #How often to print output 
        self.logStep = 10
        
        #The max number of observations to use for model selection
        self.sampleSize = 5*10**6

        # basic resultsDir
        self.resultsDir = PathDefaults.getOutputDir() + "ranking/" + dirName + "/"

        # update algoParams from command line
        self.readAlgoParams(cmdLine)
        
        #Sometimes there are problems with multiprocessing, so this fixes the issues         
        os.system('taskset -p 0xffffffff %d' % os.getpid())

    @staticmethod
    # update parameters with those from the user
    def updateParams(params, update=None):
        if update:
            for key, val in vars(update).items():
                params.__setattr__(key, val) 

    @staticmethod
    # merge default algoParameters from the class with those from the user
    def newAlgoParams(algoArgs=None):
        algoArgs_ = copy(RankingExpHelper.defaultAlgoArgs)
        RankingExpHelper.updateParams(algoArgs_, algoArgs)
        return(algoArgs_)
    
    @staticmethod
    def newAlgoParser(defaultAlgoArgs=None, add_help=False):
        # default algorithm args
        defaultAlgoArgs = RankingExpHelper.newAlgoParams(defaultAlgoArgs)
        
        # define parser
        algoParser = argparse.ArgumentParser(description="", add_help=add_help)
        for method in ["runSoftImpute", "runMaxLocalAuc"]:
            algoParser.add_argument("--" + method, action="store_true", default=defaultAlgoArgs.__getattribute__(method))
        algoParser.add_argument("--rhos", type=float, nargs="+", help="Regularisation parameter (default: %(default)s)", default=defaultAlgoArgs.rhos)
        algoParser.add_argument("--ks", type=int, nargs="+", help="Max number of singular values/vectors (default: %(default)s)", default=defaultAlgoArgs.ks)
        algoParser.add_argument("--modelSelect", action="store_true", help="Whether to do model selection on the 1st iteration (default: %(default)s)", default=defaultAlgoArgs.modelSelect)
        algoParser.add_argument("--postProcess", action="store_true", help="Whether to do post processing for soft impute (default: %(default)s)", default=defaultAlgoArgs.postProcess)
        algoParser.add_argument("--trainError", action="store_true", help="Whether to compute the error on the training matrices (default: %(default)s)", default=defaultAlgoArgs.trainError)
        algoParser.add_argument("--verbose", action="store_true", help="Whether to generate verbose algorithmic details(default: %(default)s)", default=defaultAlgoArgs.verbose)
        return(algoParser)
    
    # update current algoArgs with values from user and then from command line
    def readAlgoParams(self, cmdLine=None, defaultAlgoArgs=None):
        # update current algoArgs with values from the user
        self.__class__.updateParams(defaultAlgoArgs)
        
        # define parser, current values of algoArgs are used as default
        algoParser = self.__class__.newAlgoParser(self.algoArgs, True)

        # parse
        algoParser.parse_args(cmdLine, namespace=self.algoArgs)
            
    def printAlgoArgs(self):
        logging.info("Algo params")
        keys = list(vars(self.algoArgs).keys())
        keys.sort()
        for key in keys:
            logging.info("    " + str(key) + ": " + str(self.algoArgs.__getattribute__(key)))
                     
          
    def recordResults(self, trainX, testX, learner, fileName):
        """
        Save results for a particular recommendation 
        """
        metaData = []
        logging.debug("Computing recommendation errors")
        

        start = time.time()
        if type(learner) == SoftImpute:
            ZList = learner.learnModel(trainX)    
            U, s, V = ZList[0]
        else: 
            U, V = learner.learnModel(trainX)
        learnTime = time.time()-start 
        metaData.append(learnTime)

        trainMeasures = []
        trainMeasures.append(MCEvaluator.precisionAtK(trainX, U, V, 5))
        trainMeasures.append(MCEvaluator.precisionAtK(trainX, U, V, 10))
        trainMeasures.append(MCEvaluator.precisionAtK(trainX, U, V, 20))
        trainMeasures.append(MCEvaluator.localAUCApprox(trainX, U, V, self.algoArgs.u, self.algoArgs.numAucSamples))

        testMeasures = []
        testMeasures.append(MCEvaluator.precisionAtK(testX, U, V, 5))
        testMeasures.append(MCEvaluator.precisionAtK(testX, U, V, 10))
        testMeasures.append(MCEvaluator.precisionAtK(testX, U, V, 20))
        testMeasures.append(MCEvaluator.localAUCApprox(testX, U, V, self.algoArgs.u, self.algoArgs.numAucSamples))

        trainMeasures = numpy.array(trainMeasures)
        testMeasures = numpy.array(testMeasures)
        metaData = numpy.array(metaData)
        
        logging.debug("Train measures: " + str(trainMeasures))
        logging.debug("Test measures: " + str(testMeasures))
        numpy.savez(fileName, trainMeasures, testMeasures, metaData)
        logging.debug("Saved file as " + fileName)

    def runExperiment(self, X):
        """
        Run the selected ranking experiments and save results
        """
        logging.debug("Splitting into train and test sets")
        trainX, testX = SparseUtils.splitNnz(X, self.algoArgs.trainSplit)
        logging.debug("Train X shape and nnz: " + str(trainX.shape) + " " + str(trainX.nnz))    
        logging.debug("Test X shape and nnz: " + str(testX.shape) + " " + str(testX.nnz))
        
        if self.algoArgs.runSoftImpute:
            logging.debug("Running soft impute")
            resultsFileName = self.resultsDir + "ResultsSoftImpute.npz"
                
            fileLock = FileLock(resultsFileName)  
            
            if not fileLock.isLocked() and not fileLock.fileExists(): 
                fileLock.lock()
                
                trainX = trainX.toScipyCsc()
                testX = testX.toScipyCsc()
                
                try: 
                    rhos = numpy.array([self.algoArgs.rhos[0]])
                    learner = SoftImpute(rhos, eps=self.algoArgs.eps, k=self.algoArgs.ks[0])
                    
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking subsample of entries of size " + str(self.sampleSize))
                        modelSelectX = SparseUtils.submatrix(trainX, self.sampleSize)
                        
                        meanErrors, stdErrors = learner.modelSelect(modelSelectX)
                        
                        logging.debug("Mean errors = " + str(meanErrors))
                        logging.debug("Std errors = " + str(stdErrors))
                        
                        modelSelectFileName = resultsFileName.replace("Results", "ModelSelect") 
                        numpy.savez(modelSelectFileName, meanErrors, stdErrors)
                        logging.debug("Saved model selection grid as " + modelSelectFileName)                            
                        
                        rho = self.algoArgs.rhos[numpy.unravel_index(numpy.argmin(meanErrors), meanErrors.shape)[0]]
                        k = self.algoArgs.ks[numpy.unravel_index(numpy.argmin(meanErrors), meanErrors.shape)[1]]
                    else: 
                        rho = self.algoArgs.rhos[0]
                        k = self.algoArgs.ks[0]
                        
                    logging.debug(learner)                

                    self.recordResults(trainX, testX, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)
                
        if self.algoArgs.runMaxLocalAuc:
            logging.debug("Running max local AUC")
            resultsFileName = self.resultsDir + "ResultsMaxLocalAUC.npz"
                
            fileLock = FileLock(resultsFileName)  
            
            if not fileLock.isLocked() and not fileLock.fileExists(): 
                fileLock.lock()
                
                try: 
                    learner = MaxLocalAUC(self.algoArgs.rhos[0], self.algoArgs.ks[0], self.algoArgs.u, sigma=self.algoArgs.sigma, eps=self.algoArgs.eps, stochastic=True)
                    
                    learner.numRowSamples = self.algoArgs.numRowSamples
                    learner.numColSamples = self.algoArgs.numColSamples
                    learner.numAucSamples = self.algoArgs.numAucSamples
                    learner.initialAlg = "rand"
                    learner.recordStep = 50
                    learner.rate = "optimal"
                    learner.alpha = 0.1    
                    learner.t0 = 0.1   
                    learner.maxIterations = X.shape[0]*10   
                    learner.ks = self.algoArgs.ks
                    learner.rhos = self.algoArgs.rhos                        
                    
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking subsample of entries of size " + str(self.sampleSize))
                        modelSelectX = SparseUtils.submatrix(trainX, self.sampleSize)
                        
                        meanAucs, stdAucs = learner.modelSelect(modelSelectX)
                        
                        logging.debug("Mean local AUCs = " + str(meanAucs))
                        logging.debug("Std local AUCs = " + str(stdAucs))
                        
                        modelSelectFileName = resultsFileName.replace("Results", "ModelSelect") 
                        numpy.savez(modelSelectFileName, meanAucs, stdAucs)
                        logging.debug("Saved model selection grid as " + modelSelectFileName)                            
                    
                    logging.debug(learner)                

                    self.recordResults(trainX, testX, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)                
         
        logging.info("All done: see you around!")
