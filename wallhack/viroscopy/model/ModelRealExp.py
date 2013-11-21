"""
A script to estimate the HIV epidemic model parameters using ABC for real data.
"""
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
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

if len(sys.argv) > 2:
    i = int(sys.argv[2])
else: 
    i = 0 

FORMAT = "%(levelname)s:root:%(process)d:%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logging.debug("Number of processes: " + str(numProcesses))
logging.debug("Epidemic period index " + str(i))
numpy.set_printoptions(suppress=True, precision=4, linewidth=150)
numpy.seterr(invalid='raise')

resultsDir = PathDefaults.getOutputDir() + "viroscopy/real/" 
startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.realSimulationParams(ind=i)
N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize = HIVModelUtils.realABCParams(i)

logging.debug("Posterior sample size " + str(N))
logging.debug("Matching algorithm " + str(matchAlg))

alpha = 2
zeroVal = 0.9

logging.debug("="*10 + "Starting new simulation batch with index " + str(i) + "="*10) 
logging.debug("Total time of simulation is " + str(endDate-startDate))    

breakSize = targetGraph.subgraph(targetGraph.removedIndsAt(endDate)).size * breakScale
logging.debug("Largest acceptable graph is " + str(breakSize))

def createModel(t, matchAlg):
    """
    The parameter t is the particle index. 
    """
    #undirected = True
    #graph = HIVGraph(M, undirected)
 
    #We start with the observed graph at the start date 
    graph = targetGraph.subgraph(targetGraph.removedIndsAt(startDate)) 
    graph.addVertices(M-graph.size)
    
    p = Util.powerLawProbs(alpha, zeroVal)
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
    
    featureInds = numpy.ones(graph.vlist.getNumFeatures(), numpy.bool)
    featureInds[HIVVertices.dobIndex] = False 
    featureInds[HIVVertices.infectionTimeIndex] = False 
    featureInds[HIVVertices.hiddenDegreeIndex] = False 
    featureInds[HIVVertices.stateIndex] = False
    featureInds = numpy.arange(featureInds.shape[0])[featureInds]
    matcher = GraphMatch(matchAlg, alpha=matchAlpha, featureInds=featureInds, useWeightM=False)
    graphMetrics = HIVGraphMetrics2(targetGraph, breakSize, matcher, startDate)
    
    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates, T=float(endDate), T0=float(startDate), metrics=graphMetrics)
    model.setRecordStep(recordStep)

    return model
 
def createModelU(t): 
    return createModel(t, "U")
 
def createModelQCV(t): 
    return createModel(t, "QCV")
  
meanTheta, stdTheta, pertTheta = HIVModelUtils.estimatedRealTheta(i)

logging.debug("Using mean theta of " + str(meanTheta))
logging.debug("Using std theta of " + str(stdTheta))
logging.debug("Using perturbation std theta of " + str(pertTheta))
    
abcParams = HIVABCParameters(meanTheta, stdTheta, pertTheta)
thetaDir = resultsDir + "theta" + str(i) + "/"

if not os.path.exists(thetaDir): 
    os.mkdir(thetaDir)

#First get a quick estimate using Umeyama matching 
numUEpsilons = 5
epsilonArray = numpy.ones(numUEpsilons)*epsilon    
abcSMC = ABCSMC(epsilonArray, createModelU, abcParams, thetaDir, True, minEpsilon=minEpsilon)
abcSMC.setPosteriorSampleSize(N)
abcSMC.setNumProcesses(numProcesses)
abcSMC.batchSize = batchSize
abcSMC.maxRuns = abcMaxRuns
thetasArray = abcSMC.run()

#Now get something more precise 
epsilonArray = numpy.ones(numEpsilons)*epsilon    
abcSMC = ABCSMC(epsilonArray, createModelQCV, abcParams, thetaDir, True, minEpsilon=minEpsilon)
abcSMC.setPosteriorSampleSize(N)
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
