import numpy 
import logging 

from sandbox.util.Parameter import Parameter
from exp.viroscopy.model.HIVGraph import HIVGraph 

class HIVGraphMetrics(object): 
    def __init__(self, times):
        """
        A class to model metrics about and between HIVGraphs such as summary 
        statistics and distances. 
        
        :param times: An array of time points to compute statistics from 
        """
        
        self.times = times 

    def summary(self, graph): 
        """
        Compute a summary statistic on the input HIV graph         
        """
        Parameter.checkClass(graph, HIVGraph)
        summaryArray = numpy.zeros((self.times.shape[0], 2))

        for i in range(self.times.shape[0]): 
            t = self.times[i]
            subgraph = graph.subgraph(graph.infectedIndsAt(t))    
        
            summaryArray[i, :] = numpy.array([subgraph.getNumVertices(), subgraph.getNumEdges()])        
        
        return summaryArray
    
    def distance(self, summary1, summary2): 
        """
        Take as input two summary statistics computed on HIV graphs, and output 
        a distance metric.
        
        :param summary1: A summary statistic for a desired HIVGraph. 
        
        :param summary2: A summary statistic for a modelled HIVGraph. 
        """        
        return numpy.linalg.norm(summary1 - summary2)
        
    def shouldBreak(self, realSummary, graph, epsilon, currentTime): 
        """
        Given a summary statistic realSummary, a graph created by the model and 
        a value of epsilon return True if the distance has exceeded epsilon 
        otherwise False. 

        :param realSummary: Summary statistic computed on real data 
        
        :param graph: The HIVGraph generated by the model. 
        :type graph: `exp.viroscopy.model.HIVGraph`
        
        :param epsilon: The maximum distance to be accepted. 
        
        """
        newTimes = self.times[self.times <= currentTime]
        summaryStat = self.summary(graph)
        
        realSummary = realSummary[0:newTimes.shape[0]]
        summaryStat = HIVGraphMetrics(newTimes).summary(graph)

        dist = self.distance(summaryStat, realSummary)
        
        if dist >= epsilon:
            logging.debug("Distance is " + str(dist) +  " and epsilon is " + str(epsilon))
            return True
        else:
            return False  
            

     