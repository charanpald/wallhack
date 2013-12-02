import sys 
import numpy 
import unittest
import logging

from wallhack.viroscopy.model.HIVModelUtils import HIVModelUtils


class HIVModelUtilsTest(unittest.TestCase):
    def setUp(self):
        numpy.seterr(invalid='raise')
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(suppress=True, precision=4, linewidth=100)
        numpy.random.seed(21)
        
    def testSimulate(self): 
        #We want to see if we can get the same simulation twice 

        N, matchAlpha, breakScale, numEpsilons, epsilon, minEpsilon, matchAlg, abcMaxRuns, batchSize, pertScale = HIVModelUtils.toyABCParams()
        startDate, endDate, recordStep, M, targetGraph = HIVModelUtils.toySimulationParams(test=True)    
        breakSize = (targetGraph.subgraph(targetGraph.removedIndsAt(endDate)).size - targetGraph.subgraph(targetGraph.removedIndsAt(startDate)).size)  * breakScale 
        theta, sigmaTheta, pertTheta = HIVModelUtils.toyTheta() 
        
        model = HIVModelUtils.createModel(theta, targetGraph, startDate, endDate, recordStep, M, matchAlpha, breakSize, matchAlg)
        times, infectedIndices, removedIndices, graph, compTimes, graphMetrics = HIVModelUtils.simulate(model)
        
        numEdges = graph.getNumEdges()
        lastRemovedIndices = removedIndices[-1]
        
        #Simulate again 
        model = HIVModelUtils.createModel(theta, targetGraph, startDate, endDate, recordStep, M, matchAlpha, breakSize, matchAlg)
        times, infectedIndices, removedIndices, graph, compTimes, graphMetrics = HIVModelUtils.simulate(model)
        
        numEdges2 = graph.getNumEdges()
        lastRemovedIndices2 = removedIndices[-1]
        
        self.assertEquals(numEdges, numEdges2)
        self.assertEquals(lastRemovedIndices, lastRemovedIndices2)

if __name__ == '__main__':
    unittest.main()
