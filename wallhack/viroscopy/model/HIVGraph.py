import numpy
import os
import zipfile
import logging
from apgl.graph.CsArrayGraph import CsArrayGraph
from wallhack.viroscopy.model.HIVVertices import HIVVertices
from apgl.graph.VertexList import VertexList
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util 

"""
A HIVGraph is a Graph except that its vertices are HIVIndividuals and
there are some useful method to find the S, I and R individuals. 
"""

class HIVGraph(CsArrayGraph):
    def __init__(self, numVertices, undirected=True):
        """
        Create a graph with the specified number of vertices, and choose whether
        edges are directed. 
        """
        vList = HIVVertices(numVertices)
        super(HIVGraph, self).__init__(vList, undirected)
        
        self.endEventTime = None 

    def getSusceptibleSet(self):
        V = self.vList.getVertices(list(range(self.getNumVertices())))
        susceptibleSet = numpy.nonzero(V[:, HIVVertices.stateIndex] == HIVVertices.susceptible)[0]
        return set(susceptibleSet.tolist()) 

    def getInfectedSet(self):
        V = self.vList.getVertices(list(range(self.getNumVertices())))
        infectedSet = numpy.nonzero(V[:, HIVVertices.stateIndex] == HIVVertices.infected)[0]
        return set(infectedSet.tolist())

    def getRemovedSet(self):
        V = self.vList.getVertices(list(range(self.getNumVertices())))
        removedSet = numpy.nonzero(V[:, HIVVertices.stateIndex] == HIVVertices.removed)[0]
        return set(removedSet.tolist())

    def setRandomInfected(self, numInitialInfected, t=0.0):
        """
        Pick a number of people randomly to be infected at time t.  
        """
        Parameter.checkInt(numInitialInfected, 0, self.size)
        infectInds = numpy.random.permutation(self.size)[0:numInitialInfected]

        for i in infectInds:
            self.vlist.setInfected(i, t)
            
    def detectedNeighbours(self, vertexInd):
        """
        Return an array of the detected neighbours.
        """
        V = self.vList.getVertices(range(self.size))
        neighbours = self.neighbours(vertexInd)
        return neighbours[V[neighbours, HIVVertices.stateIndex] == HIVVertices.removed]

    @classmethod
    def load(cls, filename, vListType=None):
        """
        Load the graph object from the corresponding file. Data is loaded in a zip
        format as created using save().

        :param filename: The name of the file to load.
        :type filename: :class:`str`

        :returns: A graph corresponding to the one saved in filename.
        """
        Parameter.checkClass(filename, str)

        (path, filename) = os.path.split(filename)
        if path == "":
            path = "./"

        originalPath = os.getcwd()
        try:
            os.chdir(path)

            myzip = zipfile.ZipFile(filename + '.zip', 'r')
            myzip.extractall()
            myzip.close()

            #Deal with legacy files
            try:
                W = cls.loadMatrix(cls._wFilename)
                metaDict = Util.loadPickle(cls._metaFilename)
                if vListType == None:
                    vList = globals()[metaDict["vListType"]].load(cls._verticesFilename)
                else:
                    vList = vListType.load(cls._verticesFilename)
                undirected = metaDict["undirected"]

            except IOError:
                W = cls.loadMatrix(filename + cls._matExt)
                vList = VertexList.load(filename)
                undirected = Util.loadPickle(filename + cls._boolExt)

            graph = cls(vList.getNumVertices(), undirected)
            graph.W = W
            graph.setVertexList(vList)

            for tempFile in myzip.namelist():
                os.remove(tempFile)
        finally:
            os.chdir(originalPath)

        logging.debug("Loaded graph from file " + filename)
        return graph

    def infectedIndsAt(self, t): 
        """
        Compute the indices of this graph consisting only of all vertices 
        infected before time t
        
        :param t: The time point to consider. 
        :type t: `float`
        """
        
        vertexArray = self.getVertexList().getVertices()
        boolInds = numpy.logical_and(vertexArray[:, HIVVertices.infectionTimeIndex] <= t, vertexArray[:, HIVVertices.infectionTimeIndex] >= 0)
        boolInds2 = numpy.logical_not(numpy.logical_and(vertexArray[:, HIVVertices.detectionTimeIndex] <= t, vertexArray[:, HIVVertices.detectionTimeIndex] >= 0))
        boolInds = numpy.logical_and(boolInds, boolInds2)
        inds = numpy.arange(self.getNumVertices())[boolInds]
        
        return inds
        
    def removedIndsAt(self, t): 
        """
        Compute the indices of this graph consisting only of all vertices 
        removed before time t
        
        :param t: The time point to consider. 
        :type t: `float`
        """
        
        vertexArray = self.getVertexList().getVertices()
        inds = numpy.arange(self.getNumVertices())[numpy.logical_and(vertexArray[:, HIVVertices.detectionTimeIndex] <= t, vertexArray[:, HIVVertices.detectionTimeIndex] >= 0)]
        
        return inds

    def contactIndsAt(self, t): 
        """
        Compute the indices of this graph consisting only of all vertices 
        that have had at least 1 sexual contact before time t
        
        :param t: The time point to consider. 
        :type t: `float`
        """
        edges = self.getAllEdges()
        edgeValues = self.getEdgeValues(edges)

        inds = numpy.logical_and(edgeValues >= 0, edgeValues <=t)
        edges = edges[inds].flatten()        
        
        return numpy.unique(edges)
        

    def endTime(self): 
        """
        Return the time of the last infection or detection event. 
        """
        #vertexArray = self.getVertexList().getVertices()
        #imin = numpy.max(vertexArray[:, HIVVertices.infectionTimeIndex])
        #rmin = numpy.max(vertexArray[:, HIVVertices.detectionTimeIndex]) 
        
        #return numpy.max(numpy.array([imin, rmin]))
        return self.endEventTime
        
        