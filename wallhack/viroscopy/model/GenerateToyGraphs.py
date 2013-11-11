
import logging
import sys
import numpy
from apgl.graph import *
from apgl.util import *
from wallhack.viroscopy.model.HIVGraph import HIVGraph
from wallhack.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from wallhack.viroscopy.model.HIVRates import HIVRates
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils

"""
This is the epidemic model for the HIV spread in cuba. We repeat the simulation a number
of times and average the results. The purpose is to test the ABC model selection 
by using a known value of theta. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all='raise')
numpy.random.seed(24)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)

startDate, endDate, recordStep, M = HIVModelUtils.toySimulationParams(False, test=True)

numRepetitions = 10
undirected = True
outputDir = PathDefaults.getOutputDir() + "viroscopy/toy/"
theta, sigmaTheta, purtTheta = HIVModelUtils.toyTheta() 

graphList = []
numInfected = numpy.zeros(numRepetitions)
numRemoved = numpy.zeros(numRepetitions)

for j in range(numRepetitions):
    graph = HIVGraph(M, undirected)
    logging.debug("Created graph: " + str(graph))

    alpha = 2
    zeroVal = 0.9
    p = Util.powerLawProbs(alpha, zeroVal)
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())

    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates)
    model.setT(endDate)
    model.setRecordStep(recordStep)
    model.setParams(theta)
    
    logging.debug("Theta = " + str(theta))
    
    times, infectedIndices, removedIndices, graph = model.simulate(True)
    graphFileName = outputDir + "ToyEpidemicGraph" + str(j)
    graph.save(graphFileName)
    
    graphList.append(graph)
    numInfected[j] = len(graph.getInfectedSet())
    numRemoved[j] = len(graph.getRemovedSet())

logging.debug("Infected (mean, std): " + str((numpy.mean(numInfected), numpy.std(numInfected))))
logging.debug("Removed (mean, std): " + str((numpy.mean(numRemoved), numpy.std(numRemoved))))
logging.debug("All done.")
