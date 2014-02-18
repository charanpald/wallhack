import logging
import sys
import numpy
import scipy.stats

from apgl.graph import *
from sandbox.util import *
from wallhack.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from wallhack.viroscopy.model.HIVGraph import HIVGraph
from wallhack.viroscopy.model.HIVRates import HIVRates
from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils
from wallhack.viroscopy.model.HIVVertices import HIVVertices
from wallhack.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2
from sandbox.misc.GraphMatch import GraphMatch

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

    def profileSimulateGraphMatch(self): 
        N, matchAlpha, breakDist, purtScale = HIVModelUtils.toyABCParams()
        startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams()
        theta, stdTheta = HIVModelUtils.toyTheta()
    
        featureInds= numpy.ones(targetGraph.vlist.getNumFeatures(), numpy.bool)
        featureInds[HIVVertices.dobIndex] = False 
        featureInds[HIVVertices.infectionTimeIndex] = False 
        featureInds[HIVVertices.hiddenDegreeIndex] = False 
        featureInds[HIVVertices.stateIndex] = False 
        featureInds = numpy.arange(featureInds.shape[0])[featureInds]        
        
        #QCV is fastest and most accurate 
        #PATH is slowests but quite accurate 
        #RANK is very fast by less accurate than PATH 
        #U is fastest but least accurate         
        
        matcher = GraphMatch("QCV", alpha=matchAlpha, featureInds=featureInds, useWeightM=False)
        matcher.lambdaM = 50 
        matcher.init = "rand"
        graphMetrics = HIVGraphMetrics2(targetGraph, breakDist, matcher, float(endDate))        
        
        def run(): 
            times, infectedIndices, removedIndices, graph = HIVModelUtils.simulate(theta, startDate, endDate, recordStep, M, graphMetrics)
            print("Mean distance " + str(graphMetrics.meanDistance()))
        
        ProfileUtils.profile('run()', globals(), locals())


profiler = HIVEpidemicModelProfile()
#profiler.profileSimulate() #67.7
profiler.profileSimulateGraphMatch() 
