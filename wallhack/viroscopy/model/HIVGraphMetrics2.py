import numpy 
import logging 
import time 
from apgl.util.Parameter import Parameter
from sandbox.misc.GraphMatch import GraphMatch 

class HIVGraphMetrics2(object): 
    def __init__(self, realGraph, maxSize=100, matcher=None, T=1000):
        """
        A class to model metrics about and between HIVGraphs such as summary 
        statistics and distances. In this case we perform graph matching 
        using the QCV algorithm and other graph matching methods. 
        
        :param realGraph: The target epidemic graph 
        
        :param epsilon: The max mean objective before we break the simulation
        
        :param matcher: The graph matcher object to compute graph objective. 
        
        :param T: The end time of the simulation. If the simulation quits before T, then objective = 1.
    
        """
        
        self.objectives = [] 
        self.graphObjs = []
        self.graphSizes = []
        self.labelObjs = []
        self.times = []
        self.realGraph = realGraph
        self.maxSize = maxSize
        self.T = T 
        self.times = []
        
        if matcher == None: 
            self.matcher = GraphMatch("QCV")
        else: 
            self.matcher = matcher 
        
    def addGraph(self, graph): 
        """
        Compute the objective between this graph and the realGraph at the time 
        of the last event of this one. 
        """
        t = graph.endTime()
        subgraph = graph.subgraph(graph.removedIndsAt(t))  
        subRealGraph = self.realGraph.subgraph(self.realGraph.removedIndsAt(t))  
        self.graphSizes.append(subgraph.size)
        
        #Only add objective if the real graph has nonzero size
        if subRealGraph.size != 0 and subgraph.size <= self.maxSize: 
            permutation, distance, time = self.matcher.match(subgraph, subRealGraph)
            lastObj, lastGraphObj, lastLabelObj = self.matcher.distance(subgraph, subRealGraph, permutation, True, False, True) 
            
            self.times.append(time)
            self.objectives.append(lastObj)
            self.graphObjs.append(lastGraphObj)
            self.labelObjs.append(lastLabelObj)
            self.times.append(t) 
        else: 
            logging.debug("Not adding objective at time " + str(t) + " with simulated size " + str(subgraph.size) + " and real size " + str(subRealGraph.size))
            
    def meanObjective(self):
        """
        This is the moving average objective of the graph matches so far. 
        """
        objectives = numpy.array(self.objectives)       
        
        if objectives.shape[0]!=0: 
            weights = numpy.arange(1, objectives.shape[0]+1)
            return numpy.average(objectives, weights=weights)
        else: 
            return float("inf")
            
    def meanGraphObjective(self):
        """
        This is the mean graph objective of the graph matches so far. 
        """
        graphObjs = numpy.array(self.graphObjs)
        if graphObjs.shape[0]!=0: 
            return graphObjs.mean()
        else: 
            return float("inf")
            
    def meanLabelObjective(self):
        """
        This is the mean label objective of the graph matches so far. 
        """
        labelObjs = numpy.array(self.labelObjs)
        if labelObjs.shape[0]!=0: 
            return labelObjs.mean()
        else: 
            return float("inf")
        
    def shouldBreak(self): 
        """
        We break when the graph size exceeds a threshold 
        """
        if self.graphSizes[-1] > self.maxSize:
            logging.debug("Breaking as size has become too large: " + str(self.graphSizes[-1]) + " > " + str(self.maxSize))
            
        return self.graphSizes[-1] > self.maxSize 
        