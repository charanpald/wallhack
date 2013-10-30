"""
A script to estimate the HIV epidemic model parameters using ABC for real data.
"""
from apgl.util import *
from wallhack.viroscopy.model.HIVGraph import HIVGraph
from wallhack.viroscopy.model.HIVABCParameters import HIVABCParameters
from wallhack.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from wallhack.viroscopy.model.HIVRates import HIVRates
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils
from wallhack.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2
from wallhack.viroscopy.model.HIVVertices import HIVVertices
from sandbox.misc.GraphMatch import GraphMatch
from sandbox.predictors.ABCSMC import ABCSMC
import os
import logging
import sys
import numpy
import multiprocessing

assert False, "Must run with -O flag"

if len(sys.argv) > 1:
    numProcesses = int(sys.argv[1])
else: 
    numProcesses = multiprocessing.cpu_count()

FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logging.debug("Number of processes: " + str(numProcesses))
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)
numpy.seterr(invalid='raise')

resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/" 
startDate, endDates, numRecordSteps, M, targetGraph = HIVModelUtils.realSimulationParams()
posteriorSampleSize, matchAlpha, breakDist, pertScale = HIVModelUtils.realABCParams()

abcMaxRuns = 2500
batchSize = 50
numEpsilons = 15
epsilon = 0.8
alpha = 2
zeroVal = 0.9
eps = 0.001
matchAlg = "QCV"

logging.debug("Posterior sample size " + str(posteriorSampleSize))

for i, endDate in enumerate(endDates): 
    logging.debug("="*10 + "Starting new simulation batch with index " + str(i) + "="*10) 
    logging.debug("Total time of simulation is " + str(endDate-startDate))    
    
    def createModel(t):
        """
        The parameter t is the particle index. 
        """
        undirected = True
        graph = HIVGraph(M, undirected)
        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
        
        featureInds= numpy.ones(graph.vlist.getNumFeatures(), numpy.bool)
        featureInds[HIVVertices.dobIndex] = False 
        featureInds[HIVVertices.infectionTimeIndex] = False 
        featureInds[HIVVertices.hiddenDegreeIndex] = False 
        featureInds[HIVVertices.stateIndex] = False
        featureInds = numpy.arange(featureInds.shape[0])[featureInds]
        matcher = GraphMatch(matchAlg, alpha=matchAlpha, featureInds=featureInds, useWeightM=False)
        graphMetrics = HIVGraphMetrics2(targetGraph, breakDist, matcher, float(endDate))
        
        recordStep = (endDate-startDate)/float(numRecordSteps)
        rates = HIVRates(graph, hiddenDegSeq)
        model = HIVEpidemicModel(graph, rates, T=float(endDate), T0=float(startDate), metrics=graphMetrics)
        model.setRecordStep(recordStep)
    
        return model
        
    if i == 0: 
        meanTheta, stdTheta, pertTheta = HIVModelUtils.estimatedRealTheta()
    else:
        #Perturbations are based on the stds of the last thetas 
        pertTheta = stdTheta.copy()*pertScale
        #The prior is very loose as we do not want to bias too much to previous solutions 
        stdTheta = HIVModelUtils.estimatedRealTheta()[1]
        
        #Must clip the probabilities 
        stdTheta[1] = numpy.clip(stdTheta[1], 0, 1)
        stdTheta[5] = numpy.clip(stdTheta[5], 0, 1)
        
    logging.debug("Using mean theta of " + str(meanTheta))
    logging.debug("Using std theta of " + str(stdTheta))
    logging.debug("Using perturbation std theta of " + str(pertTheta))
        
    abcParams = HIVABCParameters(meanTheta, stdTheta, pertTheta)
    thetaDir = resultsDir + "theta" + str(i) + "/"
    
    if not os.path.exists(thetaDir): 
        os.mkdir(thetaDir)
    
    epsilonArray = numpy.ones(numEpsilons)*epsilon    
    
    os.system('taskset -p 0xffffffff %d' % os.getpid())
    
    abcSMC = ABCSMC(epsilonArray, createModel, abcParams, thetaDir, True, eps=eps)
    abcSMC.setPosteriorSampleSize(posteriorSampleSize)
    abcSMC.setNumProcesses(numProcesses)
    abcSMC.batchSize = batchSize
    abcSMC.maxRuns = abcMaxRuns
    thetasArray = abcSMC.run()
    
    meanTheta = numpy.mean(thetasArray, 0)
    stdTheta = numpy.std(thetasArray, 0)
    logging.debug(thetasArray)
    logging.debug("meanTheta=" + str(meanTheta))
    logging.debug("stdTheta=" + str(stdTheta))
    
    logging.debug("New epsilon array: " + str(abcSMC.epsilonArray))
    logging.debug("Number of ABC runs: " + str(abcSMC.numRuns))

logging.debug("All done!")
