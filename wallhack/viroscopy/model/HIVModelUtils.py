

"""
Keep some default parameters for the epidemic model. 
"""
import time 
import numpy 
import logging 
from apgl.util import Util 
from apgl.util import PathDefaults 
from apgl.graph.GraphStatistics import GraphStatistics
from wallhack.viroscopy.model.HIVGraph import HIVGraph
from wallhack.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from wallhack.viroscopy.model.HIVRates import HIVRates
from wallhack.viroscopy.model.HIVVertices import HIVVertices
from sandbox.misc.GraphMatch import GraphMatch
from wallhack.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2
from wallhack.viroscopy.HIVGraphReader import HIVGraphReader, CsvConverters

class HIVModelUtils(object):
    def __init__(self): 
        pass 
    
    @staticmethod
    def estimatedRealTheta(i):
        """
        This is taken from simulated runs using the real data 
        """
        #if i==2: 
        #    theta = numpy.array([200, 0.8, 0.2, 0.1, 0.1, 0.1]) 
        #    sigmaTheta = numpy.array([100, 0.2, 0.2, 0.1, 0.1, 0.1])
         
        theta = numpy.array([500, 0.5, 0.01, 0.1, 0.1, 0.1])
        sigmaTheta = numpy.array([400, 0.5, 0.01, 0.1, 0.1, 0.1])
        pertTheta = sigmaTheta/10
        return theta, sigmaTheta, pertTheta
  
    @staticmethod
    def toyTheta(): 
        theta = numpy.array([100, 0.9, 0.05, 0.001, 0.1, 0.005])
        sigmaTheta = numpy.array([10, 0.09, 0.05, 0.001, 0.1, 0.005])
        pertTheta = sigmaTheta/10
        return theta, sigmaTheta, pertTheta
        
    @staticmethod 
    def toySimulationParams(loadTarget=True, test=False): 
        
        if loadTarget: 
            resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/" 
            graphFile = resultsDir + "ToyEpidemicGraph0"
            targetGraph = HIVGraph.load(graphFile)        
        
        startDate = 0.0        
        endDate = 1000.0
        recordStep = 100
        M = 5000
        testPeriod = 300
        
        if test:
            endDate += testPeriod   
        
        if loadTarget: 
            return startDate, endDate, recordStep, M, targetGraph
        else: 
            return startDate, endDate, recordStep, M
        
    @staticmethod 
    def realSimulationParams(test=False, ind=0): 
        hivReader = HIVGraphReader()
        targetGraph = hivReader.readSimulationHIVGraph()
        
        testPeriod = 0.2 
        numRecordSteps = 5 
        #Note that 5% of the population is bi 
        M = targetGraph.size * 5
        #This needs to be from 1986 to 2004 
        #startDates = [CsvConverters.dateConv("01/01/1988"), CsvConverters.dateConv("01/01/1990")]
        startDates = [CsvConverters.dateConv("01/01/1990"), CsvConverters.dateConv("01/01/1992"), CsvConverters.dateConv("01/01/1998"), CsvConverters.dateConv("01/01/2002")]
        startDates = [float(i) for i in startDates]
        #endDates = [CsvConverters.dateConv("01/01/1991"), CsvConverters.dateConv("01/01/1993")]
        endDates = [CsvConverters.dateConv("01/01/1992"), CsvConverters.dateConv("01/01/1994"), CsvConverters.dateConv("01/01/1999"), CsvConverters.dateConv("01/01/2003")]
        endDates = numpy.array([float(i) for i in endDates])
        
        if test:
            numRecordSteps += numRecordSteps*testPeriod
            endDates += (endDates-startDates) * testPeriod    
        
        startDate = startDates[ind]
        endDate = endDates[ind]
        recordStep = (endDate-startDate)/float(numRecordSteps)        
        
        return startDate, endDate, recordStep, M, targetGraph, len(startDates)
    
    @staticmethod
    def realABCParams(i, test=False):
        N = 20 
        matchAlpha = 0.2 
        if test: 
            breakScale = 5.0 
        else: 
            breakScale = 1.2 
        numEpsilons = 10
        epsilon = 0.8
        minEpsilon = 0.4
        matchAlg = "QCV"
        abcMaxRuns = 1000
        batchSize = 50
        pertScale = 5
        
        return N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize, pertScale 

    @staticmethod
    def toyABCParams():
        N = 50 
        matchAlpha = 0.2 
        breakScale = 1.2 
        numEpsilons = 10
        epsilon = 0.8
        minEpsilon = 0.30
        matchAlg = "QCV"   
        abcMaxRuns = 50000
        batchSize = 50
        pertScale = 5
        
        return N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize, pertScale
   
    @staticmethod     
    def createModel(targetGraph, startDate, endDate, recordStep, M, matchAlpha, breakSize, matchAlg, theta=None): 
        alpha = 2
        zeroVal = 0.9
        numpy.random.seed(21)
        
        graph = targetGraph.subgraph(targetGraph.removedIndsAt(startDate)) 
        graph.addVertices(M-graph.size)
        logging.debug("Created graph: " + str(graph))   
        
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
        if theta != None: 
            model.setParams(theta)
                
        return model 
    
    @staticmethod     
    def simulate(model): 
        startTime = time.time()
        times, infectedIndices, removedIndices, graph =  model.simulate(True)
        simulationTime = time.time() - startTime
        
        graphMetrics = model.metrics 
        
        graphMatchTime = numpy.sum(graphMetrics.computationalTimes)
        logging.debug("Weighted objective " + str(graphMetrics.meanObjective()))
        
        return times, infectedIndices, removedIndices, graph, [simulationTime, graphMatchTime], graphMetrics
        
    @staticmethod 
    def generateStatistics(graph, times): 
        """
        For a given theta, simulate the epidemic, and then return a number of 
        relevant statistics. 
        """
        contactIndices = []
        removedIndices = []
        infectedIndices = []
        
        for t in times: 
            removedIndices.append(graph.removedIndsAt(t))
            infectedIndices.append(graph.infectedIndsAt(t))
            contactIndices.append(graph.contactIndsAt(t))

        infectedIndices = numpy.array(infectedIndices)

        V = graph.getVertexList().getVertices()
        graphStatsList = []
        
        for inds in [contactIndices, removedIndices]: 
            numVerticesArray  = numpy.array([len(x) for x in inds])
            maleArray  = numpy.array([numpy.sum(V[x, HIVVertices.genderIndex]==HIVVertices.male) for x in inds])
            femaleArray = numpy.array([numpy.sum(V[x, HIVVertices.genderIndex]==HIVVertices.female) for x in inds])
            heteroArray = numpy.array([numpy.sum(V[x, HIVVertices.orientationIndex]==HIVVertices.hetero) for x in inds])
            biArray = numpy.array([numpy.sum(V[x, HIVVertices.orientationIndex]==HIVVertices.bi) for x in inds])
            randDetectArray = numpy.array([numpy.sum(V[x, HIVVertices.detectionTypeIndex]==HIVVertices.randomDetect) for x in inds])
            conDetectArray = numpy.array([numpy.sum(V[x, HIVVertices.detectionTypeIndex]==HIVVertices.contactTrace) for x in inds])
            
            vertexArray = numpy.c_[numVerticesArray, maleArray, femaleArray, heteroArray, biArray, randDetectArray, conDetectArray]
        
            graphStats = GraphStatistics()
            graphStats = graphStats.sequenceScalarStats(graph, inds, slowStats=False)
            graphStatsList.append(graphStats)
        
        return vertexArray, infectedIndices, removedIndices, graphStatsList[0], graphStatsList[1]
    
    
    realTestPeriods = [365, 365, 365, 730]
