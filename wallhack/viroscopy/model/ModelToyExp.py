"""
A script to estimate the HIV epidemic model parameters using ABC for the toy data.
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

resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/" 
startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams()
N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize, pertScale  = HIVModelUtils.toyABCParams()

logging.debug("Total time of simulation is " + str(endDate-startDate))
logging.debug("Posterior sample size " + str(N))

alpha = 2
zeroVal = 0.9
epsilonArray = numpy.ones(numEpsilons)*epsilon   

breakSize = (targetGraph.subgraph(targetGraph.removedIndsAt(endDate)).size - targetGraph.subgraph(targetGraph.removedIndsAt(startDate)).size)  * breakScale
logging.debug("Largest acceptable graph is " + str(breakSize))

def createModel(t):
    """
    The parameter t is the particle index. 
    """
    undirected = True
    graph = HIVGraph(M, undirected)
    numpy.random.seed(21)
    p = Util.powerLawProbs(alpha, zeroVal)
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
    
    featureInds= numpy.ones(graph.vlist.getNumFeatures(), numpy.bool)
    featureInds[HIVVertices.dobIndex] = False 
    featureInds[HIVVertices.infectionTimeIndex] = False 
    featureInds[HIVVertices.hiddenDegreeIndex] = False 
    featureInds[HIVVertices.stateIndex] = False
    featureInds = numpy.arange(featureInds.shape[0])[featureInds]
    matcher = GraphMatch(matchAlg, alpha=matchAlpha, featureInds=featureInds, useWeightM=False)
    graphMetrics = HIVGraphMetrics2(targetGraph, breakSize, matcher, float(startDate))

    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates, T=float(endDate), T0=float(startDate), metrics=graphMetrics)
    model.setRecordStep(recordStep)

    return model

meanTheta, sigmaTheta, pertTheta = HIVModelUtils.toyTheta()
abcParams = HIVABCParameters(meanTheta, sigmaTheta, pertTheta)
thetaDir = resultsDir + "theta/"

if not os.path.exists(thetaDir): 
    os.mkdir(thetaDir)

logging.debug((meanTheta, sigmaTheta))

abcSMC = ABCSMC(epsilonArray, createModel, abcParams, thetaDir, True, minEpsilon=minEpsilon)
abcSMC.setPosteriorSampleSize(N)
abcSMC.batchSize = batchSize
abcSMC.maxRuns = abcMaxRuns
abcSMC.setNumProcesses(numProcesses)
abcSMC.pertScale = pertScale
thetasArray = abcSMC.run()

meanTheta = numpy.mean(thetasArray, 0)
stdTheta = numpy.std(thetasArray, 0)
logging.debug(thetasArray)
logging.debug("meanTheta=" + str(meanTheta))
logging.debug("stdTheta=" + str(stdTheta))
logging.debug("realTheta=" + str(HIVModelUtils.toyTheta()[0]))
logging.debug("New epsilon array: " + str(abcSMC.epsilonArray))
logging.debug("Number of ABC runs: " + str(abcSMC.numRuns))

logging.debug("All done!")
