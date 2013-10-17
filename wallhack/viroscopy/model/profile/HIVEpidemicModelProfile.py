import logging
import sys
import numpy
import scipy.stats

from apgl.graph import *
from apgl.util import *
from wallhack.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from wallhack.viroscopy.model.HIVGraph import HIVGraph
from wallhack.viroscopy.model.HIVRates import HIVRates
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(24)

class HIVEpidemicModelProfile():
    def __init__(self):
        #Total number of people in population
        numpy.random.seed(21)                
        assert False, "Must run with -O flag"

    def profileSimulate(self):
        startDate, endDates, numRecordSteps, M, targetGraph = HIVModelUtils.realSimulationParams()
        meanTheta, sigmaTheta = HIVModelUtils.estimatedRealTheta()
        
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
        model.setT(startDate+1000)
        model.setRecordStep(10)
        model.setParams(meanTheta)
        
        logging.debug("MeanTheta=" + str(meanTheta))

        ProfileUtils.profile('model.simulate()', globals(), locals())

profiler = HIVEpidemicModelProfile()
profiler.profileSimulate() #67.7
