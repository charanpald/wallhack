
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
import errno
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
    defaultAlgoArgs.k = 32 
    defaultAlgoArgs.ks = 2**numpy.arange(3, 8)
    defaultAlgoArgs.learningRateSelect = False
    defaultAlgoArgs.metric = "f1"
    defaultAlgoArgs.modelSelect = False
    defaultAlgoArgs.modelSelectSamples = 1000
    defaultAlgoArgs.numRecordAucSamples = 500
    defaultAlgoArgs.overwrite = False 
    defaultAlgoArgs.processes = multiprocessing.cpu_count()
    defaultAlgoArgs.recordFolds = 5
    defaultAlgoArgs.testSize = 5
    defaultAlgoArgs.u = 0.1
    defaultAlgoArgs.validationSize = 3
    defaultAlgoArgs.verbose = False
    
    #parameters for Bpr
    defaultAlgoArgs.lmbdaUserBpr = 0.1
    defaultAlgoArgs.lmbdaItemBpr = 0.1
    defaultAlgoArgs.lmbdaUsers = 2.0**-numpy.arange(-2, 10, 2)
    defaultAlgoArgs.lmbdaItems = 2.0**-numpy.arange(3, 9, 1)
    defaultAlgoArgs.maxIterationsBpr = 50
    defaultAlgoArgs.gammasBpr = 2.0**-numpy.arange(3, 7, 1)
    defaultAlgoArgs.gammaBpr = 0.01
    
    #parameters for CLiMF
    defaultAlgoArgs.gammaCLiMF = 0.002
    defaultAlgoArgs.gammasCLiMF = 2.0**-numpy.arange(5, 11, 2)
    defaultAlgoArgs.lmbdaCLiMF = 0.03
    defaultAlgoArgs.lmbdasCLiMF = 2.0**-numpy.arange(-1, 6, 2)
    defaultAlgoArgs.maxIterCLiMF = 50    
    
    #Parameters for KNN
    defaultAlgoArgs.kns = numpy.array([20]) 
    
    #Parameters for MlAuc
    defaultAlgoArgs.alpha = 0.5 
    defaultAlgoArgs.alphas = 2.0**-numpy.arange(1, 7, 0.5)
    defaultAlgoArgs.epsMlauc = 10**-6    
    defaultAlgoArgs.fullGradient = False
    defaultAlgoArgs.initialAlg = "svd"
    defaultAlgoArgs.itemExpP = 0.0 
    defaultAlgoArgs.itemExpQ = 0.0
    defaultAlgoArgs.itemFactors = False
    defaultAlgoArgs.lmbdaUMlauc = 0.0
    defaultAlgoArgs.lmbdaVMlauc = 1.0
    defaultAlgoArgs.lmbdasMlauc = 2.0**numpy.arange(-5, 3)
    defaultAlgoArgs.maxIterations = 100
    defaultAlgoArgs.normalise = True
    defaultAlgoArgs.numAucSamples = 10
    defaultAlgoArgs.numRowSamples = 30
    defaultAlgoArgs.parallelSGD = False
    defaultAlgoArgs.rate = "constant"
    defaultAlgoArgs.recordStep = 5
    defaultAlgoArgs.sampling = "uniform"
    defaultAlgoArgs.recommendSize = 5 
    defaultAlgoArgs.rhoMlauc = 0.0
    defaultAlgoArgs.rhosMlauc = numpy.array([0, 0.5, 1.0])
    defaultAlgoArgs.t0 = 1.0
    defaultAlgoArgs.t0s = 2.0**-numpy.arange(-1, 2, 1)
    defaultAlgoArgs.validationUsers = 0.2
    defaultAlgoArgs.z = 10
    
    #Parameters for SoftImpute 
    defaultAlgoArgs.epsSi = 10**-14
    defaultAlgoArgs.gamma = 0.0001
    defaultAlgoArgs.pSi = 50
    defaultAlgoArgs.qSi = 3
    defaultAlgoArgs.postProcess = False 
    defaultAlgoArgs.rhoSi = 0.1
    defaultAlgoArgs.rhosSi = numpy.linspace(1.0, 0.0, 6) 
    defaultAlgoArgs.svdAlg = "rsvd"
    
    #Parameters for WrMf 
    defaultAlgoArgs.alphaWrMf = 1
    defaultAlgoArgs.lmbdasWrMf = 2.0**-numpy.arange(1, 12, 2)
    defaultAlgoArgs.maxIterationsWrMf = 20     
        
    def __init__(self, cmdLine=None, defaultAlgoArgs = None, dirName=""):
        """ priority for default args
         - best priority: command-line value
         - middle priority: set-by-function value
         - lower priority: class value
        """
        # Parameters to choose which methods to run
        # Obtained merging default parameters from the class with those from the user
        self.algoArgs = RankingExpHelper.newAlgoParams(defaultAlgoArgs)
        

        self.ps = [1, 3, 5]
        
        #The max number of observations to use for model selection
        self.sampleSize = 5*10**6

        # basic resultsDir
        self.resultsDir = PathDefaults.getOutputDir() + "ranking/" + dirName + "/"
        
        #Create the results dir if it does not exist 
        #    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
        try:
            os.makedirs(self.resultsDir)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise

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
        algoParser.add_argument("--itemFactors", action="store_true", help="Whether to use only item factors (default: %(default)s)", default=defaultAlgoArgs.itemFactors)        
        algoParser.add_argument("--k", type=int, help="Max number of factors (default: %(default)s)", default=defaultAlgoArgs.k)
        algoParser.add_argument("--ks", type=int, nargs="+", help="Max number of factors (default: %(default)s)", default=defaultAlgoArgs.ks)
        algoParser.add_argument("--lmbdasCLiMF", type=float, nargs="+", help="Regularisation parameters (lambda) for CLiMF (default: %(default)s)", default=defaultAlgoArgs.lmbdasCLiMF)  
        algoParser.add_argument("--lmbdaVMlauc", type=float, help="Regularisation parameters (lambda) for max local AUC (default: %(default)s)", default=defaultAlgoArgs.lmbdaVMlauc)
        algoParser.add_argument("--lmbdasMlauc", type=float, nargs="+", help="Regularisation parameters for max local AUC (default: %(default)s)", default=defaultAlgoArgs.lmbdasMlauc)        
        algoParser.add_argument("--lmbdaUserBpr", type=float, help="Regularisation parameters for BPR (default: %(default)s)", default=defaultAlgoArgs.lmbdaUserBpr) 
        algoParser.add_argument("--lmbdaItemBpr", type=float, help="Regularisation parameters for BPR (default: %(default)s)", default=defaultAlgoArgs.lmbdaItemBpr)         
        algoParser.add_argument("--learningRateSelect", action="store_true", help="Whether to do learning rate selection (default: %(default)s)", default=defaultAlgoArgs.learningRateSelect)
        algoParser.add_argument("--maxIterations", type=int, help="Maximal number of iterations (default: %(default)s)", default=defaultAlgoArgs.maxIterations)
        algoParser.add_argument("--maxIterCLiMF", type=int, help="Maximal number of iterations for CLiMF algorithm (default: %(default)s)", default=defaultAlgoArgs.maxIterCLiMF)
        algoParser.add_argument("--metric", type=str, help="Validation loss metric (default: %(default)s)", default=defaultAlgoArgs.metric)
        algoParser.add_argument("--modelSelect", action="store_true", help="Whether to do model selection(default: %(default)s)", default=defaultAlgoArgs.modelSelect)
        algoParser.add_argument("--numAucSamples", type=int, help="Number of AUC samples for max local AUC (default: %(default)s)", default=defaultAlgoArgs.numAucSamples)
        algoParser.add_argument("--numRowSamples", type=int, help="Number of row samples for max local AUC (default: %(default)s)", default=defaultAlgoArgs.numRowSamples)
        algoParser.add_argument("--overwrite", action="store_true", help="Whether to overwrite results even if already computed (default: %(default)s)", default=defaultAlgoArgs.overwrite)
        algoParser.add_argument("--parallelSGD", action="store_true", help="Use parallel version of SGD (default: %(default)s)", default=defaultAlgoArgs.parallelSGD)
        algoParser.add_argument("--postProcess", action="store_true", help="Whether to do post processing for soft impute (default: %(default)s)", default=defaultAlgoArgs.postProcess)
        algoParser.add_argument("--processes", type=int, help="Number of CPU cores to use (default: %(default)s)", default=defaultAlgoArgs.processes)
        algoParser.add_argument("--rate", type=str, help="Learning rate type: either constant or optimal (default: %(default)s)", default=defaultAlgoArgs.rate)
        algoParser.add_argument("--recordStep", type=int, help="Number of iterations after which we display some partial results (default: %(default)s)", default=defaultAlgoArgs.recordStep)
        algoParser.add_argument("--sampling", type=str, help="The random sampling for max local AUC: uniform/rank/top (default: %(default)s)", default=defaultAlgoArgs.sampling)
        algoParser.add_argument("--rhoMlauc", type=float, help="The rho penalty for max local AUC (default: %(default)s)", default=defaultAlgoArgs.rhoMlauc)        
        algoParser.add_argument("--rhosMlauc", type=float, nargs="+", help="The rho penalty for max local AUC model selection (default: %(default)s)", default=defaultAlgoArgs.rhosMlauc)
        algoParser.add_argument("--t0", type=float, help="Learning rate decay for max local AUC (default: %(default)s)", default=defaultAlgoArgs.t0)
        algoParser.add_argument("--u", type=float, help="Focus on top proportion of u items (default: %(default)s)", default=defaultAlgoArgs.u)
        algoParser.add_argument("--validationSize", type=int, help="Number of items to use for validation users (default: %(default)s)", default=defaultAlgoArgs.validationSize)
        algoParser.add_argument("--validationUsers", type=float, help="Proportion of users to use for validation users (default: %(default)s)", default=defaultAlgoArgs.validationUsers)
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
        allTrainMeasures = [] 
        allTestMeasures = []
        allMetaData = []        
        
        for i in range(self.algoArgs.recordFolds):         
            metaData = []
            w = 1-self.algoArgs.u
            logging.debug("Computing recommendation errors")
            maxItems = self.ps[-1]
            
            start = time.time()
            if type(learner) == IterativeSoftImpute:
                trainIterator = iter([trainX])
                ZList = learner.learnModel(trainIterator)    
                U, s, V = ZList.next()
                U = U*s
                
                #trainX = sppy.csarray(trainX)
                #testX = sppy.csarray(testX)
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
    
            colNames = []
            trainMeasures = []
            testMeasures = []
            for p in self.ps: 
                trainMeasures.append(MCEvaluator.precisionAtK(trainOmegaPtr, trainOrderedItems, p))
                testMeasures.append(MCEvaluator.precisionAtK(testOmegaPtr, testOrderedItems, p))
                
                colNames.append("precision@" + str(p))
                
            for p in self.ps: 
                trainMeasures.append(MCEvaluator.recallAtK(trainOmegaPtr, trainOrderedItems, p))
                testMeasures.append(MCEvaluator.recallAtK(testOmegaPtr, testOrderedItems, p))
                
                colNames.append("recall@" + str(p))
               
            for p in self.ps: 
                trainMeasures.append(MCEvaluator.f1AtK(trainOmegaPtr, trainOrderedItems, p))
                testMeasures.append(MCEvaluator.f1AtK(testOmegaPtr, testOrderedItems, p))
                
                colNames.append("f1@" + str(p))           
               
            for p in self.ps: 
                trainMeasures.append(MCEvaluator.mrrAtK(trainOmegaPtr, trainOrderedItems, p))
                testMeasures.append(MCEvaluator.mrrAtK(testOmegaPtr, testOrderedItems, p))
                
                colNames.append("mrr@" + str(p))
    
            try: 
                r = SparseUtilsCython.computeR(U, V, w, self.algoArgs.numRecordAucSamples)
                trainMeasures.append(MCEvaluator.localAUCApprox(trainOmegaPtr, U, V, w, self.algoArgs.numRecordAucSamples, r=r))            
                testMeasures.append(MCEvaluator.localAUCApprox(testOmegaPtr, U, V, w, self.algoArgs.numRecordAucSamples, allArray=allOmegaPtr, r=r))
                
                w = 0.0            
                r = SparseUtilsCython.computeR(U, V, w, self.algoArgs.numRecordAucSamples)
                trainMeasures.append(MCEvaluator.localAUCApprox(trainOmegaPtr, U, V, w, self.algoArgs.numRecordAucSamples, r=r))
                testMeasures.append(MCEvaluator.localAUCApprox(testOmegaPtr, U, V, w, self.algoArgs.numRecordAucSamples, allArray=allOmegaPtr, r=r))
                
                colNames.append("LAUC@" + str(self.algoArgs.u))
                colNames.append("AUC")
            except:
                logging.debug("Could not compute AUCs")
                raise

            trainMeasures = numpy.array(trainMeasures)
            testMeasures = numpy.array(testMeasures)
            metaData = numpy.array(metaData)
            
            allTrainMeasures.append(trainMeasures)
            allTestMeasures.append(testMeasures)
            allMetaData.append(metaData)
            
        allTrainMeasures = numpy.array(allTrainMeasures)
        allTestMeasures = numpy.array(allTestMeasures)
        allMetaData = numpy.array(allMetaData)
        
        meanTrainMeasures = numpy.mean(allTrainMeasures, 0)
        meanTestMeasures = numpy.mean(allTestMeasures, 0)
        meanMetaData = numpy.mean(allMetaData, 0)
        
        logging.debug("Mean metrics")
        for i, colName in enumerate(colNames): 
            logging.debug(colName + ":" + str('%.4f' % meanTrainMeasures[i]) + "/" + str('%.4f' % meanTestMeasures[i]))
        
        numpy.savez(fileName, meanTrainMeasures, meanTestMeasures, meanMetaData, trainOrderedItems, testOrderedItems)
        logging.debug("Saved file as " + fileName)

    def runExperiment(self, X):
        """
        Run the selected ranking experiments and save results
        """
        logging.debug("Splitting into train and test sets")
        #Make sure different runs get the same train/test split 
        numpy.random.seed(21)
        m, n = X.shape
        #colProbs = (X.sum(0)+1)/float(m+1)
        #colProbs = colProbs**-self.algoArgs.itemExp 
        #colProbs = numpy.ones(n)/float(n)        
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
                modelSelectX = Sampling.sampleUsers(trainX, self.algoArgs.modelSelectSamples)
                modelSelectX = modelSelectX.toScipyCsr().tocsc()
                trainX = trainX.toScipyCsr().tocsc()
                testX = testX.toScipyCsr().tocsc()
                                
                try: 
                    learner = IterativeSoftImpute(self.algoArgs.rhoSi, eps=self.algoArgs.epsSi, k=self.algoArgs.k, svdAlg=self.algoArgs.svdAlg, postProcess=self.algoArgs.postProcess, p=self.algoArgs.pSi, q=self.algoArgs.qSi)
                    learner.folds = self.algoArgs.folds
                    learner.metric = self.algoArgs.metric
                    learner.numProcesses = self.algoArgs.processes
                    learner.recommendSize = self.algoArgs.recommendSize
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
            if self.algoArgs.itemFactors: 
                resultsFileName = self.resultsDir + "ResultsMaxLocalAUCItem.npz"
            else: 
                resultsFileName = self.resultsDir + "ResultsMaxLocalAUCUser.npz"
                
            fileLock = FileLock(resultsFileName)  
            
            if not (fileLock.isLocked() or fileLock.fileExists()) or self.algoArgs.overwrite: 
                fileLock.lock()
                
                try: 
                    learner = MaxLocalAUC(self.algoArgs.k, 1-self.algoArgs.u, lmbdaU=self.algoArgs.lmbdaUMlauc, lmbdaV=self.algoArgs.lmbdaVMlauc, eps=self.algoArgs.epsMlauc, stochastic=not self.algoArgs.fullGradient)
                    
                    learner.alpha = self.algoArgs.alpha    
                    learner.alphas = self.algoArgs.alphas
                    learner.folds = self.algoArgs.folds  
                    learner.initialAlg = self.algoArgs.initialAlg
                    learner.itemExpP = self.algoArgs.itemExpP
                    learner.itemExpQ = self.algoArgs.itemExpQ
                    learner.itemFactors = self.algoArgs.itemFactors
                    learner.ks = self.algoArgs.ks 
                    learner.lmbdas = self.algoArgs.lmbdasMlauc
                    learner.maxIterations = self.algoArgs.maxIterations  
                    learner.metric = self.algoArgs.metric 
                    learner.normalise = self.algoArgs.normalise
                    learner.numAucSamples = self.algoArgs.numAucSamples
                    learner.numProcesses = self.algoArgs.processes 
                    learner.numRowSamples = self.algoArgs.numRowSamples
                    learner.parallelSGD = self.algoArgs.parallelSGD
                    learner.rate = self.algoArgs.rate
                    learner.recommendSize = self.algoArgs.recommendSize
                    learner.recordStep = self.algoArgs.recordStep
                    learner.rho = self.algoArgs.rhoMlauc
                    learner.rhos = self.algoArgs.rhosMlauc
                    learner.sampling = self.algoArgs.sampling 
                    learner.t0 = self.algoArgs.t0    
                    learner.t0s = self.algoArgs.t0s
                    learner.validationSize = self.algoArgs.validationSize
                    learner.validationUsers = self.algoArgs.validationUsers
                    learner.z = self.algoArgs.z

                    """
                    if self.algoArgs.learningRateSelect:
                        logging.debug("Performing learning rate selection, taking sample size " + str(self.algoArgs.modelSelectSamples))
                        modelSelectX = Sampling.sampleUsers(trainX, self.algoArgs.modelSelectSamples)
                        logging.debug("Done")
                        objectives = learner.learningRateSelect(X)        
                        
                        rateSelectFileName = resultsFileName.replace("Results", "LearningRateSelect")
                        numpy.savez(rateSelectFileName, objectives)
                        logging.debug("Saved learning rate selection grid as " + rateSelectFileName) 
                    """
                    
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking sample size " + str(self.algoArgs.modelSelectSamples))
                        modelSelectX = Sampling.sampleUsers(trainX, self.algoArgs.modelSelectSamples)
                        
                        meanAucs, stdAucs = learner.modelSelect(modelSelectX)
                        #meanAucs, stdAucs = learner.modelSelectRandom(modelSelectX)
                        
                        modelSelectFileName = resultsFileName.replace("Results", "ModelSelect") 
                        numpy.savez(modelSelectFileName, meanAucs, stdAucs)
                        logging.debug("Saved model selection grid as " + modelSelectFileName)                            
                    
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
                    learner.recommendSize = self.algoArgs.recommendSize
                    learner.validationSize = self.algoArgs.validationSize
                                        
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking sample size " + str(self.algoArgs.modelSelectSamples))
                        modelSelectX = Sampling.sampleUsers(trainX, self.algoArgs.modelSelectSamples)
                        
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

                    learner = WeightedMf(self.algoArgs.k, alpha=self.algoArgs.alphaWrMf, lmbda=self.algoArgs.lmbdasWrMf[0], maxIterations=self.algoArgs.maxIterationsWrMf)
                    learner.folds = self.algoArgs.folds
                    learner.ks = self.algoArgs.ks
                    learner.lmbdas = self.algoArgs.lmbdasWrMf 
                    learner.metric = self.algoArgs.metric 
                    learner.numProcesses = self.algoArgs.processes
                    learner.numRecordAucSamples = self.algoArgs.numRecordAucSamples
                    learner.recommendSize = self.algoArgs.recommendSize
                    learner.validationSize = self.algoArgs.validationSize
                    
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
                    #trainX = trainX.toScipyCsr()
                    #testX = testX.toScipyCsr()

                    learner = BprRecommender(self.algoArgs.k, lmbdaUser=self.algoArgs.lmbdaUserBpr, lmbdaPos=self.algoArgs.lmbdaItemBpr, lmbdaNeg=self.algoArgs.lmbdaItemBpr, gamma=self.algoArgs.gammaBpr)
                    learner.folds = self.algoArgs.folds
                    learner.gammasBpr = self.algoArgs.gammasBpr
                    learner.ks = self.algoArgs.ks
                    learner.lmbdaItems = self.algoArgs.lmbdaItems
                    learner.lmbdaUsers = self.algoArgs.lmbdaUsers
                    learner.maxIterations = self.algoArgs.maxIterationsBpr
                    learner.metric = self.algoArgs.metric 
                    learner.numProcesses = self.algoArgs.processes
                    learner.recommendSize = self.algoArgs.recommendSize
                    learner.validationSize = self.algoArgs.validationSize
                    
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
                    modelSelectX = Sampling.sampleUsers(trainX, self.algoArgs.modelSelectSamples)
                    modelSelectX = scipy.sparse.csr_matrix(modelSelectX.toScipyCsr(), dtype=numpy.float64)
                    trainX = scipy.sparse.csr_matrix(trainX.toScipyCsr(), dtype=numpy.float64)
                    testX = testX.toScipyCsr()

                    learner = CLiMF(self.algoArgs.k, self.algoArgs.lmbdaCLiMF, self.algoArgs.gammaCLiMF)
                    learner.folds = self.algoArgs.folds  
                    learner.gammas = self.algoArgs.gammasCLiMF
                    learner.ks = self.algoArgs.ks 
                    learner.lmbdas = self.algoArgs.lmbdasCLiMF
                    learner.max_iters = self.algoArgs.maxIterCLiMF
                    learner.metric = self.algoArgs.metric 
                    learner.numProcesses = self.algoArgs.processes 
                    learner.numRecordAucSamples = self.algoArgs.numRecordAucSamples
                    learner.recommendSize = self.algoArgs.recommendSize
                    learner.validationSize = self.algoArgs.validationSize
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
