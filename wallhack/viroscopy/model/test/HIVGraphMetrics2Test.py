import apgl
import numpy 
import unittest
import numpy.testing as nptst 

from wallhack.viroscopy.model.HIVGraph import HIVGraph
from wallhack.viroscopy.model.HIVVertices import HIVVertices
from wallhack.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2
from sandbox.misc.GraphMatch import GraphMatch

class  HIVGraphMetrics2Test(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.set_printoptions(linewidth=100, suppress=True, precision=3)
        
        
        numVertices = 10
        self.graph = HIVGraph(numVertices)

        self.graph.vlist.setInfected(1, 0.0)
        self.graph.vlist.setDetected(1, 0.1, 0)
        self.graph.vlist.setInfected(2, 2.0)
        self.graph.vlist.setDetected(2, 2.0, 0)
        self.graph.vlist.setInfected(7, 3.0)
        self.graph.vlist.setDetected(7, 3.0, 0)
        self.graph.endEventTime = 3.0

    def testAddGraph(self): 
        epsilon = 0.12
        metrics = HIVGraphMetrics2(self.graph, epsilon)
        
        metrics.addGraph(self.graph)
  
        self.assertAlmostEquals(metrics.objectives[0], -0.834, 3)
        self.assertAlmostEquals(metrics.meanObjective(), -0.834, 3)
        
        #Start a new graph 
        #Compute distances directly 
        matcher = GraphMatch("QCV")
        graph =  HIVGraph(self.graph.size)
        dists = [] 
        metrics = HIVGraphMetrics2(self.graph, epsilon)
        
        graph.vlist.setInfected(1, 0.0)
        graph.vlist.setDetected(1, 0.1, 0)
        graph.endEventTime = 1
        metrics.addGraph(graph)
        
        
        t = graph.endTime()
        subgraph1 = graph.subgraph(graph.removedIndsAt(t))
        subgraph2 = self.graph.subgraph(graph.removedIndsAt(t)) 
        permutation, distance, time = matcher.match(subgraph1, subgraph2)
        lastObj, lastGraphObj, lastLabelObj = matcher.distance(subgraph1, subgraph2, permutation, True, False, True)
        self.assertEquals(metrics.objectives[-1], lastObj)

        self.assertFalse(metrics.shouldBreak())
        
        graph.vlist.setInfected(2, 2.0)
        graph.vlist.setDetected(2, 2.0, 0)
        metrics.addGraph(graph)
        
        t = graph.endTime()
        subgraph1 = graph.subgraph(graph.removedIndsAt(t))
        subgraph2 = self.graph.subgraph(graph.removedIndsAt(t)) 
        permutation, distance, time = matcher.match(subgraph1, subgraph2)
        lastObj, lastGraphObj, lastLabelObj = matcher.distance(subgraph1, subgraph2, permutation, True, False, True)
        self.assertEquals(metrics.objectives[-1], lastObj)   
        self.assertFalse(metrics.shouldBreak())
        
        graph.vlist.setInfected(7, 3.0)
        graph.vlist.setDetected(7, 3.0, 0)
        metrics.addGraph(graph)
        
        t = graph.endTime()
        subgraph1 = graph.subgraph(graph.removedIndsAt(t))
        subgraph2 = self.graph.subgraph(graph.removedIndsAt(t)) 
        permutation, distance, time = matcher.match(subgraph1, subgraph2)
        lastObj, lastGraphObj, lastLabelObj = matcher.distance(subgraph1, subgraph2, permutation, True, False, True)
        self.assertEquals(metrics.objectives[-1], lastObj) 
        self.assertFalse(metrics.shouldBreak())
        
        #Test case where one graph has zero size 
        graph1 = HIVGraph(10)
        graph2 = HIVGraph(10)
        
        graph1.vlist[:, HIVVertices.stateIndex] = HIVVertices.removed
        metrics = HIVGraphMetrics2(graph2, epsilon)
        metrics.addGraph(graph1)
        
        #Problem is that distance is 1 when one graph is zero
        self.assertEquals(len(metrics.objectives), 0) 

    def testMeanObjective(self): 
        metrics = HIVGraphMetrics2(self.graph, breakObj=0.7, matcher=None, T=1000)
        metrics.objectives = [0.5, 0.5, 0.5]
        
        self.assertEquals(metrics.meanObjective(), 0.5)
        
        metrics.objectives = [1.0, 0.5, 0.5]
        self.assertEquals(metrics.meanObjective(), 7.0/12)
        
        
        metrics.objectives = [1.0, 1.0, 0.5]
        self.assertEquals(metrics.meanObjective(), 3.0/4)
        
        self.assertTrue(metrics.shouldBreak())
        

if __name__ == '__main__':
    unittest.main()
