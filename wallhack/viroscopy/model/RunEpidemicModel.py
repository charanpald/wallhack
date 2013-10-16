
import logging
import sys
import numpy
from apgl.graph import *
from apgl.util import *
from wallhack.viroscopy.model.HIVGraph import HIVGraph
from wallhack.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from wallhack.viroscopy.model.HIVRates import HIVRates
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 

"""
This is the epidemic model for the HIV spread in cuba. Let's try to get an 
exponential infection. 
"""

assert False, "Must run with -O flag"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.seterr(all='raise')
numpy.random.seed(24)
numpy.set_printoptions(suppress=True, precision=4, linewidth=100)

startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.realSimulationParams()
M = 100
endDate = startDate + 1000
meanTheta, sigmaTheta = HIVModelUtils.estimatedRealTheta()
meanTheta = numpy.array([ 1,   0.1,    0.0,    0.00,         0.5,      0.1])
outputDir = PathDefaults.getOutputDir() + "viroscopy/"

undirected = True
graph = HIVGraph(M, undirected)
logging.info("Created graph: " + str(graph))

alpha = 2
zeroVal = 0.9
p = Util.powerLawProbs(alpha, zeroVal)
hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())

rates = HIVRates(graph, hiddenDegSeq)
model = HIVEpidemicModel(graph, rates)
model.setT0(startDate)
model.setT(endDate)
model.setRecordStep(recordStep)
model.setParams(meanTheta)

logging.debug("MeanTheta=" + str(meanTheta))

times, infectedIndices, removedIndices, graph = model.simulate(True)

statistics = GraphStatistics()
vertexArray, infectedIndices, removedIndices, contactGraphStats, removedGraphStats = HIVModelUtils.generateStatistics(graph, times)

numInfectedIndices = [len(x) for x in infectedIndices]

plt.figure(0)
plt.plot(times, numInfectedIndices)
plt.xlabel("Time")
plt.ylabel("Infected")

plt.figure(1)
plt.plot(times, contactGraphStats[:, statistics.numVerticesIndex])
plt.xlabel("Time")
plt.ylabel("Contact graph size")

plt.show()