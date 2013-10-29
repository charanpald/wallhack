

"""
Keep some default parameters for the epidemic model. 
"""
import numpy 
import logging 
from apgl.util import Util 
from apgl.util import PathDefaults 
from apgl.graph.GraphStatistics import GraphStatistics
from wallhack.viroscopy.model.HIVGraph import HIVGraph
from wallhack.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from wallhack.viroscopy.model.HIVRates import HIVRates
from wallhack.viroscopy.model.HIVVertices import HIVVertices
from wallhack.viroscopy.HIVGraphReader import HIVGraphReader, CsvConverters

class HIVModelUtils(object):
    def __init__(self): 
        pass 
    
    @staticmethod
    def estimatedRealTheta():
        """
        This is taken from simulated runs using the real data 
        """
        theta = numpy.array([200, 0.9, 0.5, 0.1, 0.1, 0.1])
        sigmaTheta = numpy.array([200, 0.4, 0.5, 0.1, 0.1, 0.1])
        pertTheta = sigmaTheta/10
        return theta, sigmaTheta, pertTheta
  
    @staticmethod
    def toyTheta(): 
        theta = numpy.array([100, 0.9, 0.05, 0.001, 0.1, 0.005])
        sigmaTheta = theta/10.0
        pertTheta = sigmaTheta/10
        return theta, sigmaTheta, pertTheta
        
    @staticmethod 
    def toySimulationParams(loadTarget=True): 
        
        if loadTarget: 
            resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/" 
            graphFile = resultsDir + "ToyEpidemicGraph0"
            targetGraph = HIVGraph.load(graphFile)        
        
        startDate = 0.0        
        endDate = 1000.0
        recordStep = 50
        M = 5000
        
        if loadTarget: 
            return startDate, endDate, recordStep, M, targetGraph
        else: 
            return startDate, endDate, recordStep, M
        
    @staticmethod 
    def realSimulationParams(): 
        hivReader = HIVGraphReader()
        targetGraph = hivReader.readSimulationHIVGraph()
        
        numRecordSteps = 10 
        #Note that 5% of the population is bi 
        M = targetGraph.size * 5
        #This needs to be from 1986 to 2004 
        startDate = CsvConverters.dateConv("01/01/1986")
        endDates = [CsvConverters.dateConv("01/01/1987"), CsvConverters.dateConv("01/01/1989"), CsvConverters.dateConv("01/01/1991")]
        endDates = [float(i) for i in endDates]
        
        return float(startDate), endDates, numRecordSteps, M, targetGraph
    
    @staticmethod
    def realABCParams():
        N = 50 
        matchAlpha = 0.2 
        breakDist = 0.8 
        pertScale = 0.5
        
        return N, matchAlpha, breakDist, pertScale 

    @staticmethod
    def toyABCParams():
        N = 50 
        matchAlpha = 0.2 
        breakDist = 0.8 
        pertScale = 0.1
        
        return N, matchAlpha, breakDist, pertScale  
   
    @staticmethod     
    def simulate(theta, startDate, endDate, recordStep, M, graphMetrics=None): 
        undirected = True
        graph = HIVGraph(M, undirected)
        logging.debug("Created graph: " + str(graph))
    
        alpha = 2
        zeroVal = 0.9
        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
    
        rates = HIVRates(graph, hiddenDegSeq)
        model = HIVEpidemicModel(graph, rates, endDate, startDate, metrics=graphMetrics)
        model.setRecordStep(recordStep)
        model.setParams(theta)
        
        logging.debug("Theta = " + str(theta))
        
        return model.simulate(True)
        
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
    
    toyTestPeriod = 250 
    realTestPeriods = [365, 365, 365, 730]