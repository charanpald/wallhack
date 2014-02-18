
import numpy 
import scipy.io
from apgl.graph.VertexList import VertexList
from sandbox.util.Util import Util
from sandbox.util.Parameter import Parameter 

class HIVVertices(VertexList):
    def __init__(self, numVertices):
        self.numFeatures = 8
        super(HIVVertices, self).__init__(numVertices, self.numFeatures)

        #Need to randomly set up the initial values
        self.V = self.__generateRandomVertices(numVertices)


    def __generateRandomVertices(self, n): 
        V = numpy.zeros((n, self.numFeatures))
        V[:, self.dobIndex] = numpy.random.rand(n)
        V[:, self.genderIndex] = Util.randomChoice(numpy.array([1, 1]), n)
        #Note in reality females cannot be recorded as bisexual but we model the real scenario
        #We assume that 5% of the population is gay or bisexual 
        V[:, self.orientationIndex] = Util.randomChoice(numpy.array([19, 1]), n)

        V[:, self.stateIndex] = numpy.zeros(n)
        V[:, self.infectionTimeIndex] = numpy.ones(n)*-1
        V[:, self.detectionTimeIndex] = numpy.ones(n)*-1
        V[:, self.detectionTypeIndex] = numpy.ones(n)*-1

        V[:, self.hiddenDegreeIndex] = numpy.ones(n)*-1
        
        return V 
        
    def addVertices(self, n):
        """
        We add n vertices to this object, randomly generating their features 
        """
        super(HIVVertices, self).addVertices(n)
        self.V[-n:, :] = self.__generateRandomVertices(n)

    def setInfected(self, vertexInd, time):
        Parameter.checkIndex(vertexInd, 0, self.getNumVertices())
        Parameter.checkFloat(time, 0.0, float('inf'))

        if self.V[vertexInd, HIVVertices.stateIndex] == HIVVertices.infected:
            raise ValueError("Person is already infected")

        self.V[vertexInd, HIVVertices.stateIndex] = HIVVertices.infected
        self.V[vertexInd, HIVVertices.infectionTimeIndex] = time
        

    def setDetected(self, vertexInd, time, detectionType):
        Parameter.checkIndex(vertexInd, 0, self.getNumVertices())
        Parameter.checkFloat(time, 0.0, float('inf'))

        if detectionType not in [HIVVertices.randomDetect, HIVVertices.contactTrace]:
             raise ValueError("Invalid detection type : " + str(detectionType))

        if self.V[vertexInd, HIVVertices.stateIndex] != HIVVertices.infected:
            raise ValueError("Person must be infected to be detected")

        self.V[vertexInd, HIVVertices.stateIndex] = HIVVertices.removed
        self.V[vertexInd, HIVVertices.detectionTimeIndex] = time
        self.V[vertexInd, HIVVertices.detectionTypeIndex] = detectionType


    def copy(self):
        """
        Returns a copy of this object. 
        """
        vList = HIVVertices(self.V.shape[0])
        vList.setVertices(numpy.copy(self.V))
        return vList

    def subList(self, indices):
        """
        Returns a subset of this object, indicated by the given indices.
        """
        Parameter.checkList(indices, Parameter.checkIndex, (0, self.getNumVertices()))
        vList = HIVVertices(len(indices))
        vList.setVertices(self.getVertices(indices))

        return vList 

    @staticmethod
    def load(filename):
        """
        Load this object from filename.nvl.

        :param filename: The name of the file to load.
        :type filename: :class:`str`
        """
        file = open(filename + VertexList.ext, 'rb')
        V = scipy.io.mmread(file)
        file.close()

        vList = HIVVertices(V.shape[0])
        vList.V = V

        return vList

    #Some static variables
    dobIndex = 0
    genderIndex = 1
    orientationIndex = 2

    #Time varying features
    stateIndex = 3
    infectionTimeIndex = 4
    detectionTimeIndex = 5
    detectionTypeIndex = 6
    hiddenDegreeIndex = 7

    male = 0
    female = 1
    
    hetero = 0
    bi = 1
    
    susceptible = 0
    infected = 1
    removed = 2
    randomDetect = 0
    contactTrace = 1 