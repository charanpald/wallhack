
"""
Some common functions used for the recommendation experiments 
"""
import os
import gc 
import logging
import numpy
import scipy.sparse
import argparse
import time 
import sppy
import multiprocessing
from copy import copy
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.MCEvaluatorCython import MCEvaluatorCython 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.recommendation.WeightedMf import WeightedMf
from sandbox.recommendation.WarpMf import WarpMf
from sandbox.recommendation.KNNRecommender import KNNRecommender
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute
from sandbox.recommendation.CLiMF import CLiMF
from sandbox.recommendation.BprRecommender import BprRecommender
from sandbox.util.SparseUtils import SparseUtils 
from sandbox.util.SparseUtilsCython import SparseUtilsCython 
from sandbox.util.Sampling import Sampling 
from sandbox.util.FileLock import FileLock 


class RankingExpHelper(object):
    defaultAlgoArgs = argparse.Namespace()
    
    #Which algorithm to run 
    defaultAlgoArgs.runBpr = False
    defaultAlgoArgs.runCLiMF = False
    defaultAlgoArgs.runKnn = False
    defaultAlgoArgs.runMaxLocalAuc = False
    defaultAlgoArgs.runSoftImpute = False
    defaultAlgoArgs.runWarpMf = False
    defaultAlgoArgs.runWrMf = False
    
    #General algorithm parameters 
    defaultAlgoArgs.folds = 3
    defaultAlgoArgs.k = 8 
    defaultAlgoArgs.ks = 2**numpy.arange(3, 7)
    defaultAlgoArgs.learningRateSelect = False
    defaultAlgoArgs.modelSelect = False
    defaultAlgoArgs.modelSelectSamples = 1000
    defaultAlgoArgs.numRecordAucSamples = 500
    defaultAlgoArgs.overwrite = False 
    defaultAlgoArgs.processes = multiprocessing.cpu_count()
    defaultAlgoArgs.testSize = 5
    defaultAlgoArgs.u = 0.1
    defaultAlgoArgs.validationSize = 3
    defaultAlgoArgs.verbose = False
    
    #parameters for Bpr
    defaultAlgoArgs.lmbdaUser = 0.1
    defaultAlgoArgs.lmbdaPos = 0.1
    defaultAlgoArgs.lmbdaNeg = 0.1
    defaultAlgoArgs.lmbdaUsers = 2.0**-numpy.arange(0, 10, 2)
    defaultAlgoArgs.lmbdaItems = 2.0**-numpy.arange(2, 7, 1)
    defaultAlgoArgs.maxIterationsBpr = 30
    defaultAlgoArgs.gammasBpr = 2.0**-numpy.arange(0, 5, 1)
    defaultAlgoArgs.gammaBpr = 0.1
    
    #parameters for CLiMF
    defaultAlgoArgs.gammaCLiMF = 0.002
    defaultAlgoArgs.gammasCLiMF = 2.0**-numpy.arange(5, 15, 2)
    defaultAlgoArgs.lmbdaCLiMF = 0.03
    defaultAlgoArgs.lmbdasCLiMF = 2.0**-numpy.arange(-1, 8, 2)
    defaultAlgoArgs.maxIterCLiMF = 100    
    
    #Parameters for KNN
    defaultAlgoArgs.kns = numpy.array([20]) 
    
    #Parameters for MlAuc
    defaultAlgoArgs.alpha = 0.5  
    defaultAlgoArgs.alphas = 2.0**-numpy.arange(1.0, 7.0)
    defaultAlgoArgs.epsMlauc = 10**-5    
    defaultAlgoArgs.fullGradient = False
    defaultAlgoArgs.initialAlg = "svd"
    defaultAlgoArgs.lmbdaMlauc = 0
    defaultAlgoArgs.lmbdasMlauc = 2.0**-numpy.arange(-1, 14, 2)
    defaultAlgoArgs.maxIterations = 100
    defaultAlgoArgs.numAucSamples = 10
    defaultAlgoArgs.numRowSamples = 10
    defaultAlgoArgs.rate = "optimal"
    defaultAlgoArgs.recordStep = 2
    defaultAlgoArgs.sampling = "uniform"
    defaultAlgoArgs.rhoMlauc = 0.1
    defaultAlgoArgs.rhosMlauc = 2.0**-numpy.arange(-1, 10, 2)
    defaultAlgoArgs.t0 = 10**-3 
    defaultAlgoArgs.t0s = 2.0**-numpy.arange(0.0, 6.0)
    
    #Parameters for SoftImpute 
    defaultAlgoArgs.epsSi = 10**-14
    defaultAlgoArgs.gamma = 0.0001
    defaultAlgoArgs.postProcess = False 
    defaultAlgoArgs.rhoSi = 0.1
    defaultAlgoArgs.rhosSi = numpy.linspace(0.5, 0.0, 6) 
    defaultAlgoArgs.svdAlg = "rsvd"
    
    #Parameters for WrMf 
    defaultAlgoArgs.alphaWrMf = 1
    defaultAlgoArgs.lmbdasWrMf = 2.0**-numpy.arange(1, 12, 2)
    defaultAlgoArgs.numIterationsWrMf = 20     
        
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
        for method in ["runSoftImpute", "runMaxLocalAuc", "runWarpMf", "runWrMf", "runKnn", "runCLiMF", "runBpr"]:
            algoParser.add_argument("--" + method, action="store_true", default=defaultAlgoArgs.__getattribute__(method))
        algoParser.add_argument("--alpha", type=float, help="Learning rate for max local AUC (default: %(default)s)", default=defaultAlgoArgs.alpha)
        algoParser.add_argument("--fullGradient", action="store_true", help="Whether to compute the full gradient at each iteration (default: %(default)s)", default=defaultAlgoArgs.fullGradient)
        algoParser.add_argument("--gamma", type=float, help="Regularisation parameter (gamma) for CLiMF (default: %(default)s)", default=defaultAlgoArgs.gamma)     
        algoParser.add_argument("--folds", type=int, help="Folds/repetitions for model selection (default: %(default)s)", default=defaultAlgoArgs.folds)   
        algoParser.add_argument("--initialAlg", type=str, help="Initial setup for U and V for max local AUC: either rand or svd (default: %(default)s)", default=defaultAlgoArgs.initialAlg)
        algoParser.add_argument("--ks", type=int, nargs="+", help="Max number of singular values/vectors (default: %(default)s)", default=defaultAlgoArgs.ks)
        algoParser.add_argument("--lmbdasCLiMF", type=float, nargs="+", help="Regularisation parameters (lambda) for CLiMF (default: %(default)s)", default=defaultAlgoArgs.lmbdasCLiMF)        
        algoParser.add_argument("--lmbdasMlauc", type=float, nargs="+", help="Regularisation parameters for max local AUC (default: %(default)s)", default=defaultAlgoArgs.lmbdasMlauc)        
        algoParser.add_argument("--learningRateSelect", action="store_true", help="Whether to do learning rate selection (default: %(default)s)", default=defaultAlgoArgs.learningRateSelect)
        algoParser.add_argument("--maxIterations", type=int, help="Maximal number of iterations (default: %(default)s)", default=defaultAlgoArgs.maxIterations)
        algoParser.add_argument("--maxIterCLiMF", type=int, help="Maximal number of iterations for CLiMF algorithm (default: %(default)s)", default=defaultAlgoArgs.maxIterCLiMF)
        algoParser.add_argument("--modelSelect", action="store_true", help="Whether to do model selection(default: %(default)s)", default=defaultAlgoArgs.modelSelect)
        algoParser.add_argument("--numAucSamples", type=int, help="Number of AUC samples for max local AUC (default: %(default)s)", default=defaultAlgoArgs.numAucSamples)
        algoParser.add_argument("--numRowSamples", type=int, help="Number of row samples for max local AUC (default: %(default)s)", default=defaultAlgoArgs.numRowSamples)
        algoParser.add_argument("--overwrite", action="store_true", help="Whether to overwrite results even if already computed (default: %(default)s)", default=defaultAlgoArgs.overwrite)
        algoParser.add_argument("--postProcess", action="store_true", help="Whether to do post processing for soft impute (default: %(default)s)", default=defaultAlgoArgs.postProcess)
        algoParser.add_argument("--processes", type=int, help="Number of CPU cores to use (default: %(default)s)", default=defaultAlgoArgs.processes)
        algoParser.add_argument("--rate", type=str, help="Learning rate type: either constant or optimal (default: %(default)s)", default=defaultAlgoArgs.rate)
        algoParser.add_argument("--recordStep", type=int, help="Number of iterations after which we display some partial results (default: %(default)s)", default=defaultAlgoArgs.recordStep)
        algoParser.add_argument("--sampling", type=str, help="The random sampling for max local AUC: uniform/rank/top (default: %(default)s)", default=defaultAlgoArgs.sampling)
        algoParser.add_argument("--rhoMlauc", type=float, help="The rho penalty for max local AUC (default: %(default)s)", default=defaultAlgoArgs.rhoMlauc)        
        algoParser.add_argument("--rhosMlauc", type=float, nargs="+", help="The rho penalty for max local AUC model selection (default: %(default)s)", default=defaultAlgoArgs.rhosMlauc)
        algoParser.add_argument("--t0", type=float, help="Learning rate decay for max local AUC (default: %(default)s)", default=defaultAlgoArgs.t0)
        algoParser.add_argument("--u", type=float, help="Focus on top proportion of u items (default: %(default)s)", default=defaultAlgoArgs.u)
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
                     
          
    def recordResults(self, X, trainX, testX, learner, fileName):
        """
        Save results for a particular recommendation 
        """
        metaData = []
        w = 1-self.algoArgs.u
        logging.debug("Computing recommendation errors")
        ps = [1, 3, 5]
        maxItems = ps[-1]
        

        start = time.time()
        if type(learner) == IterativeSoftImpute:
            trainIterator = iter([trainX])
            ZList = learner.learnModel(trainIterator)    
            U, s, V = ZList.next()
            U = U*s
            
            trainX = sppy.csarray(trainX)
            testX = sppy.csarray(testX)
            U = numpy.ascontiguousarray(U)
            V = numpy.ascontiguousarray(V)
        else: 
            learner.learnModel(trainX)
            U = learner.U 
            V = learner.V 
            
        learnTime = time.time()-start 
        metaData.append(learnTime)

        logging.debug("Getting all omega")
        allOmegaPtr = SparseUtils.getOmegaListPtr(X)
        logging.debug("Getting train omega")
        trainOmegaPtr = SparseUtils.getOmegaListPtr(trainX)
        logging.debug("Getting test omega")
        testOmegaPtr = SparseUtils.getOmegaListPtr(testX)
        logging.debug("Getting recommendations")
                
        trainOrderedItems = MCEvaluator.recommendAtk(U, V, maxItems)
        testOrderedItems = MCEvaluatorCython.recommendAtk(U, V, maxItems, trainX)


        trainMeasures = []
        testMeasures = []
        for p in ps: 
            
            trainMeasures.append(MCEvaluator.precisionAtK(trainOmegaPtr, trainOrderedItems, p))
            testMeasures.append(MCEvaluator.precisionAtK(testOmegaPtr, testOrderedItems, p))
            
            logging.debug("precision@" + str(p) + " (train/test):" + str(trainMeasures[-1]) + str("/") + str(testMeasures[-1]))
            
        for p in ps: 
            trainMeasures.append(MCEvaluator.recallAtK(trainOmegaPtr, trainOrderedItems, p))
            testMeasures.append(MCEvaluator.recallAtK(testOmegaPtr, testOrderedItems, p))
            
            logging.debug("recall@" + str(p) + " (train/test):" + str(trainMeasures[-1]) + str("/") + str(testMeasures[-1]))
            

        try: 
            r = SparseUtilsCython.computeR(U, V, w, self.algoArgs.numRecordAucSamples)
            trainMeasures.append(MCEvaluator.localAUCApprox(trainOmegaPtr, U, V, w, self.algoArgs.numRecordAucSamples, r=r))            
            testMeasures.append(MCEvaluator.localAUCApprox(testOmegaPtr, U, V, w, self.algoArgs.numRecordAucSamples, allArray=allOmegaPtr, r=r))
            
            w = 0.0            
            r = SparseUtilsCython.computeR(U, V, w, self.algoArgs.numRecordAucSamples)
            trainMeasures.append(MCEvaluator.localAUCApprox(trainOmegaPtr, U, V, w, self.algoArgs.numRecordAucSamples, r=r))
            testMeasures.append(MCEvaluator.localAUCApprox(testOmegaPtr, U, V, w, self.algoArgs.numRecordAucSamples, allArray=allOmegaPtr, r=r))
            
            logging.debug("Local AUC@" + str(self.algoArgs.u) +  " (train/all):" + str(trainMeasures[-2]) + str("/") + str(testMeasures[-2]))
            logging.debug("AUC (train/test):" + str(trainMeasures[-1]) + str("/") + str(testMeasures[-1]))
        except:
            logging.debug("Could not compute AUCs")
            raise

        trainMeasures = numpy.array(trainMeasures)
        testMeasures = numpy.array(testMeasures)
        metaData = numpy.array(metaData)
        
        numpy.savez(fileName, trainMeasures, testMeasures, metaData, trainOrderedItems, testOrderedItems)
        logging.debug("Saved file as " + fileName)

    def runExperiment(self, X):
        """
        Run the selected ranking experiments and save results
        """
        logging.debug("Splitting into train and test sets")
        trainTestXs = Sampling.shuffleSplitRows(X, 1, self.algoArgs.testSize)
        trainX, testX = trainTestXs[0]
        logging.debug("Train X shape and nnz: " + str(trainX.shape) + " " + str(trainX.nnz))    
        logging.debug("Test X shape and nnz: " + str(testX.shape) + " " + str(testX.nnz))
        
        if self.algoArgs.runSoftImpute:
            logging.debug("Running soft impute")
            resultsFileName = self.resultsDir + "ResultsSoftImpute.npz"
                
            fileLock = FileLock(resultsFileName)  
            
            if not (fileLock.isLocked() or fileLock.fileExists()) or self.algoArgs.overwrite:
                fileLock.lock()
                modelSelectX = trainX[0:self.algoArgs.modelSelectSamples, :]
                modelSelectX = modelSelectX.toScipyCsr().tocsc()
                trainX = trainX.toScipyCsr().tocsc()
                testX = testX.toScipyCsr().tocsc()
                                
                try: 
                    learner = IterativeSoftImpute(self.algoArgs.rhoSi, eps=self.algoArgs.epsSi, k=self.algoArgs.k, svdAlg=self.algoArgs.svdAlg, postProcess=self.algoArgs.postProcess)
                    learner.numProcesses = self.algoArgs.processes
                    learner.folds = self.algoArgs.folds
                    learner.metric = "precision"
                    learner.validationSize = self.algoArgs.validationSize
                    
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking subsample of entries of size " + str(self.sampleSize))
                        
                        cvInds = Sampling.randCrossValidation(self.algoArgs.folds, modelSelectX.nnz)
                        meanErrors, stdErrors = learner.modelSelect2(modelSelectX, self.algoArgs.rhosSi, self.algoArgs.ks, cvInds)
                        
                        modelSelectFileName = resultsFileName.replace("Results", "ModelSelect") 
                        numpy.savez(modelSelectFileName, meanErrors, stdErrors)
                        logging.debug("Saved model selection grid as " + modelSelectFileName)                            
                        
                    logging.debug(learner)                

                    self.recordResults(X, trainX, testX, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)
                
        if self.algoArgs.runMaxLocalAuc:
            logging.debug("Running max local AUC")
            resultsFileName = self.resultsDir + "ResultsMaxLocalAUC.npz"
                
            fileLock = FileLock(resultsFileName)  
            
            if not (fileLock.isLocked() or fileLock.fileExists()) or self.algoArgs.overwrite: 
                fileLock.lock()
                
                try: 
                    learner = MaxLocalAUC(self.algoArgs.k, 1-self.algoArgs.u, lmbda=self.algoArgs.lmbdaMlauc, eps=self.algoArgs.epsMlauc, stochastic=not self.algoArgs.fullGradient)
                    
                    learner.numRowSamples = self.algoArgs.numRowSamples
                    learner.numAucSamples = self.algoArgs.numAucSamples
                    learner.initialAlg = self.algoArgs.initialAlg
                    learner.recordStep = self.algoArgs.recordStep
                    learner.rate = self.algoArgs.rate
                    learner.alpha = self.algoArgs.alpha    
                    learner.t0 = self.algoArgs.t0    
                    learner.maxIterations = self.algoArgs.maxIterations  
                    learner.ks = self.algoArgs.ks 
                    learner.folds = self.algoArgs.folds  
                    learner.numProcesses = self.algoArgs.processes 
                    learner.lmbdas = self.algoArgs.lmbdasMlauc
                    learner.rho = self.algoArgs.rhoMlauc
                    learner.rhos = self.algoArgs.rhosMlauc
                    learner.validationSize = self.algoArgs.validationSize
                    learner.alphas = self.algoArgs.alphas
                    learner.t0s = self.algoArgs.t0s
                    learner.metric = "precision"
                    learner.sampling = self.algoArgs.sampling 

                    if self.algoArgs.learningRateSelect:
                        logging.debug("Performing learning rate selection, taking sample size " + str(self.algoArgs.modelSelectSamples))
                        modelSelectX = trainX[0:self.algoArgs.modelSelectSamples, :]
                        logging.debug("Done")
                        objectives = learner.learningRateSelect(modelSelectX)        
                        
                        rateSelectFileName = resultsFileName.replace("Results", "LearningRateSelect")
                        numpy.savez(rateSelectFileName, objectives)
                        logging.debug("Saved learning rate selection grid as " + rateSelectFileName) 
                    
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking sample size " + str(self.algoArgs.modelSelectSamples))
                        modelSelectX = trainX[0:self.algoArgs.modelSelectSamples, :]
                        
                        meanAucs, stdAucs = learner.modelSelect(modelSelectX)
                        
                        modelSelectFileName = resultsFileName.replace("Results", "ModelSelect") 
                        numpy.savez(modelSelectFileName, meanAucs, stdAucs)
                        logging.debug("Saved model selection grid as " + modelSelectFileName)                            
                    
                    learner.maxIterations = self.algoArgs.maxIterations*2 
                    logging.debug(learner)                

                    self.recordResults(X, trainX, testX, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)                
        
        if self.algoArgs.runWarpMf: 
            logging.debug("Running WARP loss MF")
            resultsFileName = self.resultsDir + "ResultsWarpMf.npz"
                
            fileLock = FileLock(resultsFileName)     
            
            if not (fileLock.isLocked() or fileLock.fileExists()) or self.algoArgs.overwrite:
                fileLock.lock()
                
                try: 
                    trainX = trainX.toScipyCsr()
                    testX = testX.toScipyCsr()

                    learner = WarpMf(self.algoArgs.k, self.algoArgs.lmbdas[0], u=self.algoArgs.u)
                    learner.ks = self.algoArgs.ks
                    learner.numProcesses = self.algoArgs.processes
                    learner.validationSize = self.algoArgs.validationSize
                                        
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking sample size " + str(self.algoArgs.modelSelectSamples))
                        modelSelectX = trainX[0:self.algoArgs.modelSelectSamples, :]
                        
                        meanAucs, stdAucs = learner.modelSelect(modelSelectX)
                        
                        logging.debug("Mean local AUCs = " + str(meanAucs))
                        logging.debug("Std local AUCs = " + str(stdAucs))
                        
                        modelSelectFileName = resultsFileName.replace("Results", "ModelSelect")
                        numpy.savez(modelSelectFileName, meanAucs, stdAucs)
                        logging.debug("Saved model selection grid as " + modelSelectFileName)                        
                    
                    logging.debug(learner)   
                    
                    self.recordResults(X, trainX, testX, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)         

        if self.algoArgs.runWrMf: 
            logging.debug("Running Weighted Regularized Matrix Factorization")
            resultsFileName = self.resultsDir + "ResultsWrMf.npz"
                
            fileLock = FileLock(resultsFileName)     
            
            if not (fileLock.isLocked() or fileLock.fileExists()) or self.algoArgs.overwrite: 
                fileLock.lock()
                
                try: 
                    trainX = trainX.toScipyCsr()
                    testX = testX.toScipyCsr()

                    learner = WeightedMf(self.algoArgs.k, alpha=self.algoArgs.alphaWrMf, lmbda=self.algoArgs.lmbdasWrMf[0], numIterations=self.algoArgs.numIterationsWrMf, w=1-self.algoArgs.u)
                    learner.ks = self.algoArgs.ks
                    learner.lmbdas = self.algoArgs.lmbdasWrMf 
                    learner.numProcesses = self.algoArgs.processes
                    learner.validationSize = self.algoArgs.validationSize
                    learner.folds = self.algoArgs.folds
                    learner.numRecordAucSamples = self.algoArgs.numRecordAucSamples
                    
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking sample size " + str(self.algoArgs.modelSelectSamples))
                        modelSelectX = trainX[0:min(trainX.shape[0], self.algoArgs.modelSelectSamples), :]
                        
                        meanAucs, stdAucs = learner.modelSelect(modelSelectX)
                        
                        modelSelectFileName = resultsFileName.replace("Results", "ModelSelect") 
                        numpy.savez(modelSelectFileName, meanAucs, stdAucs)
                        logging.debug("Saved model selection grid as " + modelSelectFileName)                            
                    
                    logging.debug(learner)   
                    
                    self.recordResults(X, trainX, testX, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)  
       
        if self.algoArgs.runBpr: 
            logging.debug("Running Bayesian Personalised Recommendation")
            resultsFileName = self.resultsDir + "ResultsBpr.npz"
                
            fileLock = FileLock(resultsFileName)     
            
            if not (fileLock.isLocked() or fileLock.fileExists()) or self.algoArgs.overwrite: 
                fileLock.lock()
                
                try: 
                    trainX = trainX.toScipyCsr()
                    testX = testX.toScipyCsr()

                    learner = BprRecommender(self.algoArgs.k, lmbdaUser=self.algoArgs.lmbdaUser, lmbdaPos=self.algoArgs.lmbdaPos, lmbdaNeg=self.algoArgs.lmbdaNeg, gamma=self.algoArgs.gammaBpr, w=1-self.algoArgs.u)
                    learner.ks = self.algoArgs.ks
                    learner.numProcesses = self.algoArgs.processes
                    learner.validationSize = self.algoArgs.validationSize
                    learner.folds = self.algoArgs.folds
                    learner.lmbdaUsers = self.algoArgs.lmbdaUsers
                    learner.lmbdaItems = self.algoArgs.lmbdaItems
                    learner.gammasBpr = self.algoArgs.gammasBpr
                    learner.maxIterations = self.algoArgs.maxIterationsBpr
                    
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking sample size " + str(self.algoArgs.modelSelectSamples))
                        modelSelectX = trainX[0:min(trainX.shape[0], self.algoArgs.modelSelectSamples), :]
                        
                        meanAucs, stdAucs = learner.modelSelect(modelSelectX)
                        
                        modelSelectFileName = resultsFileName.replace("Results", "ModelSelect") 
                        numpy.savez(modelSelectFileName, meanAucs, stdAucs)
                        logging.debug("Saved model selection grid as " + modelSelectFileName)                            
                    
                    logging.debug(learner)   
                    
                    self.recordResults(X, trainX, testX, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)         
       
        if self.algoArgs.runKnn: 
            logging.debug("Running kNN")
            resultsFileName = self.resultsDir + "ResultsKnn.npz"
                
            fileLock = FileLock(resultsFileName)     
            
            if not (fileLock.isLocked() or fileLock.fileExists()) or self.algoArgs.overwrite:
                fileLock.lock()
                
                try: 
                    trainX = trainX.toScipyCsr()
                    testX = testX.toScipyCsr()

                    learner = KNNRecommender(self.algoArgs.kns[0])
                    learner.numProcesses = self.algoArgs.processes
                         
                    logging.debug(learner)   
                    
                    self.recordResults(X, trainX, testX, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)         

        if self.algoArgs.runCLiMF: 
            # !!!! no model selection
            logging.debug("Running CLiMF")
            resultsFileName = self.resultsDir + "ResultsCLiMF.npz"
                
            fileLock = FileLock(resultsFileName)
            
            if not (fileLock.isLocked() or fileLock.fileExists()) or self.algoArgs.overwrite:
                fileLock.lock()
                
                try: 
                    modelSelectX = trainX[0:self.algoArgs.modelSelectSamples, :]
                    modelSelectX = scipy.sparse.csr_matrix(modelSelectX.toScipyCsr(), dtype=numpy.float64)
                    trainX = scipy.sparse.csr_matrix(trainX.toScipyCsr(), dtype=numpy.float64)
                    testX = testX.toScipyCsr()

                    learner = CLiMF(self.algoArgs.k, self.algoArgs.lmbdaCLiMF, self.algoArgs.gammaCLiMF)
                    learner.max_iters = self.algoArgs.maxIterCLiMF
                    learner.ks = self.algoArgs.ks 
                    learner.gammas = self.algoArgs.gammasCLiMF
                    learner.lmbdas = self.algoArgs.lmbdasCLiMF
                    learner.numRecordAucSamples = self.algoArgs.numRecordAucSamples
                    learner.w = 1-self.algoArgs.u
                    learner.folds = self.algoArgs.folds  
                    learner.validationSize = self.algoArgs.validationSize
                    learner.numProcesses = self.algoArgs.processes 
                    learner.verbose = self.algoArgs.verbose

                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking sample size " + str(self.algoArgs.modelSelectSamples))

                        meanObjs, stdObjs = learner.modelSelect(modelSelectX)
                        
                        modelSelectFileName = resultsFileName.replace("Results", "ModelSelect") 
                        numpy.savez(modelSelectFileName, meanObjs, stdObjs)
                        logging.debug("Saved model selection grid as " + modelSelectFileName)                            
                    
                    logging.debug(learner)                
                    
                    self.recordResults(X, trainX, testX, learner, resultsFileName)
                finally: 
                    fileLock.unlock()
            else: 
                logging.debug("File is locked or already computed: " + resultsFileName)         
       
        logging.info("All done: see you around!")
