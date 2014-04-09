
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
import multiprocessing
from copy import copy
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.recommendation.MaxLocalAUC import MaxLocalAUC
from sandbox.recommendation.WeightedMf import WeightedMf
from sandbox.recommendation.WarpMf import WarpMf
from sandbox.recommendation.KNNRecommender import KNNRecommender
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute
from sandbox.util.SparseUtils import SparseUtils 
from sandbox.util.Sampling import Sampling 
from sandbox.util.FileLock import FileLock 


class RankingExpHelper(object):
    defaultAlgoArgs = argparse.Namespace()
    defaultAlgoArgs.alpha = 0.1
    defaultAlgoArgs.epsSi = 10**-14
    defaultAlgoArgs.epsMlauc = 10**-6
    defaultAlgoArgs.folds = 4
    defaultAlgoArgs.fullGradient = False
    defaultAlgoArgs.initialAlg = "svd"
    defaultAlgoArgs.ks = 2**numpy.arange(3, 8)
    defaultAlgoArgs.kns = numpy.array([20])
    defaultAlgoArgs.learningRateSelect = False
    defaultAlgoArgs.lmbdasWrMf = 2.0**-numpy.arange(1, 12, 2)
    defaultAlgoArgs.lmbdasMlauc = 2.0**-numpy.arange(1, 12, 2)
    defaultAlgoArgs.maxIterations = 5000
    defaultAlgoArgs.modelSelect = False
    defaultAlgoArgs.nu = 20
    defaultAlgoArgs.numAucSamples = 20
    defaultAlgoArgs.numRecordAucSamples = 500
    defaultAlgoArgs.numRowSamples = 20
    defaultAlgoArgs.numStepIterations = 1000
    defaultAlgoArgs.overwrite = False 
    defaultAlgoArgs.postProcess = False 
    defaultAlgoArgs.processes = multiprocessing.cpu_count()
    defaultAlgoArgs.rate = "optimal"
    defaultAlgoArgs.recordStep = defaultAlgoArgs.numStepIterations 
    defaultAlgoArgs.rhoMlauc = 0.000
    defaultAlgoArgs.rhos = numpy.linspace(0.5, 0.0, 6) 
    defaultAlgoArgs.runKnn = False
    defaultAlgoArgs.runMaxLocalAuc = False
    defaultAlgoArgs.runSoftImpute = False
    defaultAlgoArgs.runWarpMf = False
    defaultAlgoArgs.runWrMf = False
    defaultAlgoArgs.sigma = 0.2
    defaultAlgoArgs.t0 = 10**-4 
    defaultAlgoArgs.trainSplit = 0.8
    defaultAlgoArgs.u = 0.1
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
        for method in ["runSoftImpute", "runMaxLocalAuc", "runWarpMf", "runWrMf", "runKnn"]:
            algoParser.add_argument("--" + method, action="store_true", default=defaultAlgoArgs.__getattribute__(method))
        algoParser.add_argument("--alpha", type=float, help="Learning rate for max local AUC (default: %(default)s)", default=defaultAlgoArgs.alpha)
        algoParser.add_argument("--fullGradient", action="store_true", help="Whether to compute the full gradient at each iteration (default: %(default)s)", default=defaultAlgoArgs.fullGradient)
        algoParser.add_argument("--initialAlg", type=str, help="Initial setup for U and V for max local AUC: either rand or svd (default: %(default)s)", default=defaultAlgoArgs.initialAlg)
        algoParser.add_argument("--ks", type=int, nargs="+", help="Max number of singular values/vectors (default: %(default)s)", default=defaultAlgoArgs.ks)
        algoParser.add_argument("--lmbdasMlauc", type=float, nargs="+", help="Regularisation parameters for max local AUC (default: %(default)s)", default=defaultAlgoArgs.lmbdasMlauc)        
        algoParser.add_argument("--learningRateSelect", action="store_true", help="Whether to do learning rate selection (default: %(default)s)", default=defaultAlgoArgs.learningRateSelect)
        algoParser.add_argument("--maxIterations", type=int, help="Maximal number of iterations (default: %(default)s)", default=defaultAlgoArgs.maxIterations)
        algoParser.add_argument("--modelSelect", action="store_true", help="Whether to do model selection(default: %(default)s)", default=defaultAlgoArgs.modelSelect)
        algoParser.add_argument("--numAucSamples", type=int, help="Number of AUC samples for max local AUC (default: %(default)s)", default=defaultAlgoArgs.numAucSamples)
        algoParser.add_argument("--numRowSamples", type=int, help="Number of row samples for max local AUC (default: %(default)s)", default=defaultAlgoArgs.numRowSamples)
        algoParser.add_argument("--nu", type=int, help="Weight of discordance for max local AUC (default: %(default)s)", default=defaultAlgoArgs.nu)
        algoParser.add_argument("--overwrite", action="store_true", help="Whether to overwrite results even if already computed (default: %(default)s)", default=defaultAlgoArgs.overwrite)
        algoParser.add_argument("--postProcess", action="store_true", help="Whether to do post processing for soft impute (default: %(default)s)", default=defaultAlgoArgs.postProcess)
        algoParser.add_argument("--processes", type=int, help="Number of CPU cores to use (default: %(default)s)", default=defaultAlgoArgs.processes)
        algoParser.add_argument("--rate", type=str, help="Learning rate type: either constant or optimal (default: %(default)s)", default=defaultAlgoArgs.rate)
        algoParser.add_argument("--recordStep", type=int, help="Number of iterations after which we display some partial results (default: %(default)s)", default=defaultAlgoArgs.recordStep)
        algoParser.add_argument("--rhoMlauc", type=float, help="The penalisation on non-orthogonal columns for U, V for max local AUC (default: %(default)s)", default=defaultAlgoArgs.rhoMlauc)        
        algoParser.add_argument("--rhos", type=float, nargs="+", help="Regularisation parameter for SoftImpute (default: %(default)s)", default=defaultAlgoArgs.rhos)
        algoParser.add_argument("--sigma", type=int, help="Learning rate for (stochastic) gradient descent (default: %(default)s)", default=defaultAlgoArgs.sigma)
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
        ps = [3, 5, 10, 20]
        maxItems = ps[-1]
        

        start = time.time()
        if type(learner) == IterativeSoftImpute:
            trainIterator = iter([trainX])
            ZList = learner.learnModel(trainIterator)    
            U, s, V = ZList.next()
            U = U*s
            
            trainX = sppy.csarray(trainX)
            testX = sppy.csarray(testX)
        else: 
            learner.learnModel(trainX)
        learnTime = time.time()-start 
        metaData.append(learnTime)


        logging.debug("Getting all omega")
        omegaList = SparseUtils.getOmegaList(X)
        logging.debug("Getting train omega")
        trainOmegaList = SparseUtils.getOmegaList(trainX)
        logging.debug("Getting test omega")
        testOmegaList = SparseUtils.getOmegaList(testX)
        logging.debug("Getting recommendations")
                
        
        if type(learner) == IterativeSoftImpute:
            orderedItems = MCEvaluator.recommendAtk(U, V, maxItems)
        else: 
            orderedItems = learner.predict(maxItems)

        trainMeasures = []
        testMeasures = []
        for p in ps: 
            trainMeasures.append(MCEvaluator.precisionAtK(trainX, orderedItems, p, omegaList=trainOmegaList))
            testMeasures.append(MCEvaluator.precisionAtK(testX, orderedItems, p, omegaList=testOmegaList))
            
            logging.debug("precision@" + str(p) + " (train/test/total):" + str(trainMeasures[-1]) + str("/") + str(testMeasures[-1])+ str("/") + str(trainMeasures[-1]+testMeasures[-1]))
            
        for p in ps: 
            trainMeasures.append(MCEvaluator.recallAtK(trainX, orderedItems, p, omegaList=trainOmegaList))
            testMeasures.append(MCEvaluator.recallAtK(testX, orderedItems, p, omegaList=testOmegaList))
            
            logging.debug("recall@" + str(p) + " (train/test):" + str(trainMeasures[-1]) + str("/") + str(testMeasures[-1]))
            

        try: 
            if type(learner) != IterativeSoftImpute:
                U = learner.U 
                V = learner.V 
            
            trainMeasures.append(MCEvaluator.localAUCApprox(trainX, U, V, w, self.algoArgs.numRecordAucSamples, omegaList=trainOmegaList))
            trainMeasures.append(MCEvaluator.localAUCApprox(trainX, U, V, 0.0, self.algoArgs.numRecordAucSamples, omegaList=trainOmegaList))
            testMeasures.append(MCEvaluator.localAUCApprox(X, U, V, w, self.algoArgs.numRecordAucSamples, omegaList=omegaList))
            testMeasures.append(MCEvaluator.localAUCApprox(X, U, V, 0.0, self.algoArgs.numRecordAucSamples, omegaList=omegaList))
            
            logging.debug("Local AUC@" + str(self.algoArgs.u) +  " (train/all):" + str(trainMeasures[-2]) + str("/") + str(testMeasures[-2]))
            logging.debug("Local AUC@1 (train/all):" + str(trainMeasures[-1]) + str("/") + str(testMeasures[-1]))
        except:
            logging.debug("Could not compute AUCs")
            raise

        trainMeasures = numpy.array(trainMeasures)
        testMeasures = numpy.array(testMeasures)
        metaData = numpy.array(metaData)
        
        numpy.savez(fileName, trainMeasures, testMeasures, metaData, orderedItems)
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
            
            if not (fileLock.isLocked() or fileLock.fileExists()) or self.algoArgs.overwrite:
                fileLock.lock()
                
                #print(trainX.storagetype)
                trainX = trainX.toScipyCsr().tocsc()
                testX = testX.toScipyCsr().tocsc()
                                
                try: 
                    learner = IterativeSoftImpute(self.algoArgs.rhos[0], eps=self.algoArgs.epsSi, k=self.algoArgs.ks[0], svdAlg="propack")
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
                    learner = MaxLocalAUC(self.algoArgs.ks[0], 1-self.algoArgs.u, lmbda=self.algoArgs.lmbdasMlauc[0], sigma=self.algoArgs.sigma, eps=self.algoArgs.epsMlauc, stochastic=not self.algoArgs.fullGradient)
                    
                    learner.numRowSamples = self.algoArgs.numRowSamples
                    learner.numAucSamples = self.algoArgs.numAucSamples
                    learner.nu = self.algoArgs.nu
                    learner.initialAlg = self.algoArgs.initialAlg
                    learner.recordStep = self.algoArgs.recordStep
                    learner.rate = self.algoArgs.rate
                    learner.alpha = self.algoArgs.alpha    
                    learner.t0 = self.algoArgs.t0    
                    learner.maxIterations = self.algoArgs.maxIterations  
                    learner.ks = self.algoArgs.ks 
                    learner.folds = self.algoArgs.folds  
                    learner.numProcesses = self.algoArgs.processes 
                    learner.numStepIterations = self.algoArgs.numStepIterations
                    learner.lmbdas = self.algoArgs.lmbdasMlauc
                    learner.rho = self.algoArgs.rhoMlauc

                    if self.algoArgs.learningRateSelect:
                        logging.debug("Performing learning rate selection, taking subsample of entries of size " + str(self.sampleSize))
                        modelSelectX = SparseUtils.submatrix(trainX, self.sampleSize)
                        objectives = learner.learningRateSelect(modelSelectX)        
                        
                        rateSelectFileName = resultsFileName.replace("Results", "LearningRateSelect")
                        numpy.savez(rateSelectFileName, objectives)
                        logging.debug("Saved learning rate selection grid as " + rateSelectFileName) 
                    
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking subsample of entries of size " + str(self.sampleSize))
                        modelSelectX = SparseUtils.submatrix(trainX, self.sampleSize)
                        
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
        
        if self.algoArgs.runWarpMf: 
            logging.debug("Running WARP loss MF")
            resultsFileName = self.resultsDir + "ResultsWarpMf.npz"
                
            fileLock = FileLock(resultsFileName)     
            
            if not (fileLock.isLocked() or fileLock.fileExists()) or self.algoArgs.overwrite:
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

                    learner = WeightedMf(self.algoArgs.ks[0], self.algoArgs.lmbdasWrMf[0], u=self.algoArgs.u)
                    learner.ks = self.algoArgs.ks
                    learner.lmbdas = self.algoArgs.lmbdasWrMf 
                    learner.numProcesses = self.algoArgs.processes
                    
                    if self.algoArgs.modelSelect: 
                        logging.debug("Performing model selection, taking subsample of entries of size " + str(self.sampleSize))
                        modelSelectX = SparseUtils.submatrix(trainX, self.sampleSize)
                        
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
       
        logging.info("All done: see you around!")
