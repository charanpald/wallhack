
"""
Some common functions used for the recommendation experiments 
"""
import os
import gc 
import logging
import numpy
import argparse
import time 
import sppy
from copy import copy
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.recommendation.WeightedMf import WeightedMf
from sandbox.recommendation.WarpMf import WarpMf
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute
from sandbox.util.SparseUtils import SparseUtils 
from sandbox.util.Sampling import Sampling 
from sandbox.util.FileLock import FileLock 


class RankingExpHelper(object):
    defaultAlgoArgs = argparse.Namespace()
    defaultAlgoArgs.runMaxLocalAuc = False
    defaultAlgoArgs.runSoftImpute = False
    defaultAlgoArgs.runWarpMf = False
    defaultAlgoArgs.runWrMf = False
    defaultAlgoArgs.rhos = numpy.linspace(0.5, 0.0, 6) 
    defaultAlgoArgs.lmbdas = numpy.linspace(0.5, 0.0, 6)     
    defaultAlgoArgs.folds = 4
    defaultAlgoArgs.u = 0.1
    defaultAlgoArgs.eps = 0.01
    defaultAlgoArgs.sigma = 0.2
    defaultAlgoArgs.numRowSamples = 50
    defaultAlgoArgs.numColSamples = 50
    defaultAlgoArgs.numAucSamples = 100
    defaultAlgoArgs.nu = 20
    defaultAlgoArgs.nuBar = 1
    defaultAlgoArgs.maxIterations = 1000
    defaultAlgoArgs.trainSplit = 2.0/3
    defaultAlgoArgs.modelSelect = False
    defaultAlgoArgs.learningRateSelect = False
    defaultAlgoArgs.postProcess = False 
    defaultAlgoArgs.trainError = False 
    defaultAlgoArgs.verbose = False
    defaultAlgoArgs.processes = 8
    defaultAlgoArgs.fullGradient = False
    defaultAlgoArgs.rate = "optimal"
    defaultAlgoArgs.recordStep = 50 
    defaultAlgoArgs.initialAlg = "rand"
    
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
        for method in ["runSoftImpute", "runMaxLocalAuc", "runWarpMf", "runWrMf"]:
            algoParser.add_argument("--" + method, action="store_true", default=defaultAlgoArgs.__getattribute__(method))
        algoParser.add_argument("--rhos", type=float, nargs="+", help="Regularisation parameter (default: %(default)s)", default=defaultAlgoArgs.rhos)
        algoParser.add_argument("--ks", type=int, nargs="+", help="Max number of singular values/vectors (default: %(default)s)", default=defaultAlgoArgs.ks)
        algoParser.add_argument("--modelSelect", action="store_true", help="Whether to do model selection(default: %(default)s)", default=defaultAlgoArgs.modelSelect)
        algoParser.add_argument("--postProcess", action="store_true", help="Whether to do post processing for soft impute (default: %(default)s)", default=defaultAlgoArgs.postProcess)
        algoParser.add_argument("--verbose", action="store_true", help="Whether to generate verbose algorithmic details(default: %(default)s)", default=defaultAlgoArgs.verbose)
        algoParser.add_argument("--numRowSamples", type=int, help="Number of row samples for max local AUC (default: %(default)s)", default=defaultAlgoArgs.numRowSamples)
        algoParser.add_argument("--numColSamples", type=int, help="Number of col samples for max local AUC (default: %(default)s)", default=defaultAlgoArgs.numColSamples)
        algoParser.add_argument("--numAucSamples", type=int, help="Number of AUC samples for max local AUC (default: %(default)s)", default=defaultAlgoArgs.numAucSamples)
        algoParser.add_argument("--nu", type=int, help="Weight of discordance for max local AUC (default: %(default)s)", default=defaultAlgoArgs.nu)
        algoParser.add_argument("--nuBar", type=int, help="Weight of score threshold max local AUC (default: %(default)s)", default=defaultAlgoArgs.nuBar)
        algoParser.add_argument("--sigma", type=int, help="Learning rate for (stochastic) gradient descent (default: %(default)s)", default=defaultAlgoArgs.sigma)
        algoParser.add_argument("--recordStep", type=int, help="Number of iterations after which we display some partial results (default: %(default)s)", default=defaultAlgoArgs.recordStep)
        algoParser.add_argument("--processes", type=int, help="Number of CPU cores to use (default: %(default)s)", default=defaultAlgoArgs.processes)
        algoParser.add_argument("--maxIterations", type=int, help="Maximal number of iterations (default: %(default)s)", default=defaultAlgoArgs.maxIterations)
        algoParser.add_argument("--rate", type=str, help="Learning rate type: either constant or optimal (default: %(default)s)", default=defaultAlgoArgs.rate)
        algoParser.add_argument("--fullGradient", action="store_true", help="Whether to compute the full gradient at each iteration (default: %(default)s)", default=defaultAlgoArgs.fullGradient)
        algoParser.add_argument("--learningRateSelect", action="store_true", help="Whether to do learning rate selection (default: %(default)s)", default=defaultAlgoArgs.learningRateSelect)
                
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
        if type(learner) == IterativeSoftImpute:
            trainIterator = iter([trainX])
            ZList = learner.learnModel(trainIterator)    
            U, s, V = ZList.next()
            U = U*s
            
            trainX = sppy.csarray(trainX)
            testX = sppy.csarray(testX)
        else: 
            U, V = learner.learnModel(trainX)
        learnTime = time.time()-start 
        metaData.append(learnTime)

        logging.debug("Getting train omega")
        trainOmegaList = SparseUtils.getOmegaList(trainX)
        logging.debug("Getting test omega")
        testOmegaList = SparseUtils.getOmegaList(testX)
        logging.debug("Getting recommendations")
        scoreInds = MCEvaluator.recommendAtk(U, V, 20)

        trainMeasures = []
        trainMeasures.append(MCEvaluator.precisionAtK(trainX, U, V, 5, scoreInds=scoreInds, omegaList=trainOmegaList))
        trainMeasures.append(MCEvaluator.precisionAtK(trainX, U, V, 10, scoreInds=scoreInds, omegaList=trainOmegaList))
        trainMeasures.append(MCEvaluator.precisionAtK(trainX, U, V, 20, scoreInds=scoreInds, omegaList=trainOmegaList))
        trainMeasures.append(MCEvaluator.localAUCApprox(trainX, U, V, self.algoArgs.u, self.algoArgs.numAucSamples, omegaList=trainOmegaList))
        trainMeasures.append(MCEvaluator.localAUCApprox(trainX, U, V, 1.0, self.algoArgs.numAucSamples, omegaList=trainOmegaList))

        testMeasures = []
        testMeasures.append(MCEvaluator.precisionAtK(testX, U, V, 5, scoreInds=scoreInds, omegaList=testOmegaList))
        testMeasures.append(MCEvaluator.precisionAtK(testX, U, V, 10, scoreInds=scoreInds, omegaList=testOmegaList))
        testMeasures.append(MCEvaluator.precisionAtK(testX, U, V, 20, scoreInds=scoreInds, omegaList=testOmegaList))
        testMeasures.append(MCEvaluator.localAUCApprox(testX, U, V, self.algoArgs.u, self.algoArgs.numAucSamples, omegaList=testOmegaList))
        testMeasures.append(MCEvaluator.localAUCApprox(testX, U, V, 1.0, self.algoArgs.numAucSamples, omegaList=testOmegaList))

        trainMeasures = numpy.array(trainMeasures)
        testMeasures = numpy.array(testMeasures)
        metaData = numpy.array(metaData)
        
        logging.debug("Train measures: " + str(trainMeasures))
        logging.debug("Test measures: " + str(testMeasures))
        numpy.savez(fileName, trainMeasures, testMeasures, metaData, scoreInds)
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
                
                #print(trainX.storagetype)
                trainX = trainX.toScipyCsr().tocsc()
                testX = testX.toScipyCsr().tocsc()
                                
                try: 
                    learner = IterativeSoftImpute(self.algoArgs.rhos[0], eps=self.algoArgs.eps, k=self.algoArgs.ks[0], svdAlg="rsvd")
                    learner.numProcesses = self.algoArgs.processes
                    
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking subsample of entries of size " + str(self.sampleSize))
                        modelSelectX = SparseUtils.submatrix(trainX, self.sampleSize)
                        
                        cvInds = Sampling.randCrossValidation(self.algoArgs.folds, modelSelectX.nnz)
                        meanErrors, stdErrors = learner.modelSelect(modelSelectX, self.algoArgs.rhos, self.algoArgs.ks, cvInds)
                        
                        logging.debug("Mean errors = " + str(meanErrors))
                        logging.debug("Std errors = " + str(stdErrors))
                        
                        modelSelectFileName = resultsFileName.replace("Results", "ModelSelect") 
                        numpy.savez(modelSelectFileName, meanErrors, stdErrors)
                        logging.debug("Saved model selection grid as " + modelSelectFileName)                            
                        
                    logging.debug(learner)                

                    self.recordResults(trainX, testX, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)
                
        if self.algoArgs.runMaxLocalAuc:
            logging.debug("Running max local AUC")
            resultsFileName = self.resultsDir + "ResultsMaxLocalAUC_nrs="+str(self.algoArgs.numRowSamples)+"_ncs="+str(self.algoArgs.numColSamples)+"_nas="+str(self.algoArgs.numAucSamples)
            resultsFileName += "_nu=" +str(self.algoArgs.nu)+"_nuBar="+str(self.algoArgs.nuBar)+".npz"
                
            fileLock = FileLock(resultsFileName)  
            
            if not fileLock.isLocked() and not fileLock.fileExists(): 
                fileLock.lock()
                
                try: 
                    learner = MaxLocalAUC(self.algoArgs.rhos[0], self.algoArgs.ks[0], self.algoArgs.u, sigma=self.algoArgs.sigma, eps=self.algoArgs.eps, stochastic=not self.algoArgs.fullGradient)
                    
                    learner.numRowSamples = self.algoArgs.numRowSamples
                    learner.numColSamples = self.algoArgs.numColSamples
                    learner.numAucSamples = self.algoArgs.numAucSamples
                    learner.nu = self.algoArgs.nu
                    learner.nuBar = self.algoArgs.nuBar
                    learner.initialAlg = self.algoArgs.initialAlg
                    learner.recordStep = self.algoArgs.recordStep
                    learner.rate = self.algoArgs.rate
                    learner.alpha = 100    
                    learner.t0 = 0.001   
                    learner.maxIterations = self.algoArgs.maxIterations  
                    learner.ks = self.algoArgs.ks
                    learner.rhos = self.algoArgs.rhos   
                    learner.folds = self.algoArgs.folds  
                    learner.numProcesses = self.algoArgs.processes 

                    if self.algoArgs.learningRateSelect:
                        logging.debug("Performing learning rate selection, taking subsample of entries of size " + str(self.sampleSize))
                        modelSelectX = SparseUtils.submatrix(trainX, self.sampleSize)
                        objectives = learner.learningRateSelect(modelSelectX)        
                        
                        logging.debug("Objectives = " + str(objectives))
                        
                        rateSelectFileName = resultsFileName.replace("Results", "LearningRateSelect")
                        numpy.savez(rateSelectFileName, objectives)
                        logging.debug("Saved learning rate selection grid as " + rateSelectFileName) 
                    
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
        
        if self.algoArgs.runWarpMf: 
            logging.debug("Running WARP loss MF")
            resultsFileName = self.resultsDir + "ResultsWarpMf.npz"
                
            fileLock = FileLock(resultsFileName)     
            
            if not fileLock.isLocked() and not fileLock.fileExists(): 
                fileLock.lock()
                
                try: 
                    trainX = trainX.toScipyCsr()
                    testX = testX.toScipyCsr()

                    learner = WarpMf(self.algoArgs.ks[0], self.algoArgs.lmbdas[0], u=self.algoArgs.u)
                    learner.ks = self.algoArgs.ks
                    learner.numProcesses = self.algoArgs.processes
                                        
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

        if self.algoArgs.runWrMf: 
            logging.debug("Running Weighted Regularized Matrix Factorization")
            resultsFileName = self.resultsDir + "ResultsWrMf.npz"
                
            fileLock = FileLock(resultsFileName)     
            
            if not fileLock.isLocked() and not fileLock.fileExists(): 
                fileLock.lock()
                
                try: 
                    trainX = trainX.toScipyCsr()
                    testX = testX.toScipyCsr()

                    learner = WeightedMf(self.algoArgs.ks[0], self.algoArgs.lmbdas[0], u=self.algoArgs.u)
                    learner.ks = self.algoArgs.ks
                    learner.numProcesses = self.algoArgs.processes
                    
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
