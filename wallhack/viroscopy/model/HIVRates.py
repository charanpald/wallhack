import numpy
from wallhack.viroscopy.model.HIVVertices import HIVVertices
from apgl.util.Util import *
import numpy.testing as nptst
from random import choice 

"""
Model the contact rate of an infected individual and other susceptibles.
"""
class HIVRates():
    def __init__(self, graph, hiddenDegSeq):
        """
        Graph is the initial HIV graph and hiddenDegSeq is the initial degree
        sequence. 
        """
        #Trade off between choosing new contact from known degrees and random is q-p for real degree 
        #against p for the hidden degree
        self.p = 1
        self.q = 2

        self.graph = graph

        #First figure out the different types of people in the graph
        self.femaleInds = self.graph.vlist.V[:, HIVVertices.genderIndex]==HIVVertices.female
        self.maleInds = self.graph.vlist.V[:, HIVVertices.genderIndex]==HIVVertices.male
        self.biMaleInds = numpy.logical_and(self.maleInds, self.graph.vlist.V[:, HIVVertices.orientationIndex]==HIVVertices.bi)
        self.heteroMaleInds = numpy.logical_and(self.maleInds, self.graph.vlist.V[:, HIVVertices.orientationIndex]==HIVVertices.hetero)
        self.biFemaleInds = numpy.logical_and(self.femaleInds, self.graph.vlist.V[:, HIVVertices.orientationIndex]==HIVVertices.bi)

        #We need to store degree sequences for 4 types
        #Each expanded degree sequence is a list of indices repeated according to the degree of the corresponding vertex 
        #We add 1 to the hidden degree sequence and degree sequence of the graph 
        degreeCounts = numpy.zeros(self.graph.size, numpy.int)
        degreeCounts[self.femaleInds] = 1
        degreeCounts[self.femaleInds] += graph.outDegreeSequence()[self.femaleInds]*(self.q-self.p)
        degreeCounts[self.femaleInds] += hiddenDegSeq[self.femaleInds]*self.p
        self.expandedDegSeqFemales = Util.expandIntArray(degreeCounts)
        
        degreeCounts = numpy.zeros(self.graph.size, numpy.int)
        degreeCounts[self.maleInds] = 1
        degreeCounts[self.maleInds] += graph.outDegreeSequence()[self.maleInds]*(self.q-self.p)
        degreeCounts[self.maleInds] += hiddenDegSeq[self.maleInds]*self.p
        self.expandedDegSeqMales = Util.expandIntArray(degreeCounts)      
        
        degreeCounts = numpy.zeros(self.graph.size, numpy.int)
        degreeCounts[self.biMaleInds] = 1
        degreeCounts[self.biMaleInds] += graph.outDegreeSequence()[self.biMaleInds]*(self.q-self.p)
        degreeCounts[self.biMaleInds] += hiddenDegSeq[self.biMaleInds]*self.p
        self.expandedDegSeqBiMales = Util.expandIntArray(degreeCounts)          
        
        degreeCounts = numpy.zeros(self.graph.size, numpy.int)
        degreeCounts[self.biFemaleInds] = 1
        degreeCounts[self.biFemaleInds] += graph.outDegreeSequence()[self.biFemaleInds]*(self.q-self.p)
        degreeCounts[self.biFemaleInds] += hiddenDegSeq[self.biFemaleInds]*self.p
        self.expandedDegSeqBiFemales = Util.expandIntArray(degreeCounts)  
        
        nptst.assert_array_equal(numpy.unique(self.expandedDegSeqFemales), numpy.arange(self.graph.size)[self.femaleInds]) 
        nptst.assert_array_equal(numpy.unique(self.expandedDegSeqMales), numpy.arange(self.graph.size)[self.maleInds])
        nptst.assert_array_equal(numpy.unique(self.expandedDegSeqBiMales), numpy.arange(self.graph.size)[self.biMaleInds])
        nptst.assert_array_equal(numpy.unique(self.expandedDegSeqBiFemales), numpy.arange(self.graph.size)[self.biFemaleInds])

        #Check degree sequence         
        if __debug__: 
            binShape = numpy.bincount(self.expandedDegSeqFemales).shape[0]
            assert (numpy.bincount(self.expandedDegSeqFemales)[self.femaleInds[0:binShape]] == 
                (graph.outDegreeSequence()*(self.q-self.p)+hiddenDegSeq*self.p)[self.femaleInds[0:binShape]] + 1).all()
              
            binShape = numpy.bincount(self.expandedDegSeqMales).shape[0]
            assert (numpy.bincount(self.expandedDegSeqMales)[self.maleInds[0:binShape]] == 
                (graph.outDegreeSequence()*(self.q-self.p)+hiddenDegSeq*self.p)[self.maleInds[0:binShape]] + 1).all()
                
            if self.expandedDegSeqBiMales.shape[0]!=0:
                binShape = numpy.bincount(self.expandedDegSeqBiMales).shape[0]                
                assert (numpy.bincount(self.expandedDegSeqBiMales)[self.biMaleInds[0:binShape]] == 
                    (graph.outDegreeSequence()*(self.q-self.p)+hiddenDegSeq*self.p)[self.biMaleInds[0:binShape]] + 1).all()
                    
            if self.expandedDegSeqBiFemales.shape[0]!=0:
                binShape = numpy.bincount(self.expandedDegSeqBiFemales).shape[0]                
                assert (numpy.bincount(self.expandedDegSeqBiFemales)[self.biFemaleInds[0:binShape]] == 
                    (graph.outDegreeSequence()*(self.q-self.p)+hiddenDegSeq*self.p)[self.biFemaleInds[0:binShape]] + 1).all()

        self.hiddenDegSeq = hiddenDegSeq
        self.degSequence = graph.outDegreeSequence() 

        #Parameters for sexual contact
        self.alpha = 0.5 
        self.contactRate = 0.5

        #Infection probabilities are from wikipedia
        self.infectProb = 50.0/10000

        #Random detection
        self.randDetectRate = 1/720.0

        #Contact tracing parameters 
        self.ctRatePerPerson = 0.3
        #The start and end time of contact tracing
        self.ctStartTime = 180
        self.ctEndTime = 1825

        #contactTimesArr is an array of the index of the last sexual contact or -1 
        #if no previous contact 
        self.previousContact = numpy.ones(graph.getNumVertices())*-1
        self.neighboursList = []
        self.detectedNeighboursList = [] 

        for i in range(graph.size):
            self.neighboursList.append(graph.neighbours(i))
            self.detectedNeighboursList.append(numpy.array([], numpy.int))
            
            #For a graph with edges choose the previous contact randomly 
            if len(self.neighboursList[-1]) != 0: 
                self.previousContact[i] = choice(self.neighboursList[-1])    

    def setAlpha(self, alpha):
        Parameter.checkFloat(alpha, 0.0, 1.0)
        
        if alpha == 0: 
            raise ValueError("Alpha must be greater than zero")
        
        self.alpha = alpha

    def setContactRate(self, contactRate):
        Parameter.checkFloat(contactRate, 0.0, float('inf'))
        self.contactRate = contactRate

    def setRandDetectRate(self, randDetectRate):
        Parameter.checkFloat(randDetectRate, 0.0, float('inf'))
        self.randDetectRate = randDetectRate

    def setCtRatePerPerson(self, ctRatePerPerson):
        Parameter.checkFloat(ctRatePerPerson, 0.0, float('inf'))
        self.ctRatePerPerson = ctRatePerPerson

    def setInfectProb(self, infectProb):
        Parameter.checkFloat(infectProb, 0.0, 1.0)
        self.infectProb = infectProb
        
    #@profile
    def contactEvent(self, vertexInd1, vertexInd2, t):
        """
        Indicates a sexual contact event between two vertices. 
        """
        if vertexInd1 == vertexInd2:
            return 
        if self.graph.getEdge(vertexInd1, vertexInd2) == None:
            for i in [vertexInd1, vertexInd2]:
                self.degSequence[i] += 1 
                if self.graph.vlist.V[i, HIVVertices.genderIndex] == HIVVertices.male:
                    self.expandedDegSeqMales = numpy.append(self.expandedDegSeqMales, numpy.repeat(numpy.array([i]), self.q-self.p))

                    if self.graph.vlist.V[i, HIVVertices.orientationIndex]==HIVVertices.bi:
                        self.expandedDegSeqBiMales = numpy.append(self.expandedDegSeqBiMales, numpy.repeat(numpy.array([i]), self.q-self.p))
                else:
                    self.expandedDegSeqFemales = numpy.append(self.expandedDegSeqFemales, numpy.repeat(numpy.array([i]), self.q-self.p))
                    
                    if self.graph.vlist.V[i, HIVVertices.orientationIndex]==HIVVertices.bi:
                        self.expandedDegSeqBiFemales = numpy.append(self.expandedDegSeqBiFemales, numpy.repeat(numpy.array([i]), self.q-self.p))
           
        if __debug__: 
            inds = numpy.unique(self.expandedDegSeqMales)
            assert (self.graph.vlist.V[inds, HIVVertices.genderIndex] == HIVVertices.male).all()
            
            inds = numpy.unique(self.expandedDegSeqFemales)
            assert (self.graph.vlist.V[inds, HIVVertices.genderIndex] == HIVVertices.female).all()
            
            inds = numpy.unique(self.expandedDegSeqBiMales)
            assert (numpy.logical_and(self.graph.vlist.V[inds, HIVVertices.genderIndex] == HIVVertices.male, self.graph.vlist.V[inds, HIVVertices.orientationIndex] == HIVVertices.bi)).all()
            
            inds = numpy.unique(self.expandedDegSeqBiFemales)
            assert (numpy.logical_and(self.graph.vlist.V[inds, HIVVertices.genderIndex] == HIVVertices.female, self.graph.vlist.V[inds, HIVVertices.orientationIndex] == HIVVertices.bi)).all()
           
        self.graph.addEdge(vertexInd1, vertexInd2, t)
        self.neighboursList[vertexInd1] = self.graph.neighbours(vertexInd1)
        self.neighboursList[vertexInd2] = self.graph.neighbours(vertexInd2)
        self.previousContact[vertexInd1] = vertexInd2
        self.previousContact[vertexInd2] = vertexInd1

        assert (self.degSequence == self.graph.outDegreeSequence()).all()

        #assert self.expandedDegSeq.shape[0] == numpy.sum(self.graph.outDegreeSequence()) + self.graph.getNumVertices(), \
        #    "expandedDegSequence.shape[0]=%d, sum(degreeSequence)=%d" % (self.expandedDegSeq.shape[0], self.graph.getNumVertices()+numpy.sum(self.graph.outDegreeSequence()))

    def removeEvent(self, vertexInd, detectionMethod, t):
        """
        We just remove the vertex from expandedDegSeq and expandedHiddenDegSeq
        """
        self.graph.vlist.setDetected(vertexInd, t, detectionMethod)

        #Note that we don't remove the neighbour because he/she can still be a contact 
        #Therefore this is the degree sequence minus removed vertices 
        if self.graph.vlist.V[vertexInd, HIVVertices.genderIndex] == HIVVertices.male:
            self.expandedDegSeqMales = self.expandedDegSeqMales[self.expandedDegSeqMales!=vertexInd]
            
            if self.graph.vlist.V[vertexInd, HIVVertices.orientationIndex]==HIVVertices.bi:
                self.expandedDegSeqBiMales = self.expandedDegSeqBiMales[self.expandedDegSeqBiMales!=vertexInd]    
        else: 
            self.expandedDegSeqFemales = self.expandedDegSeqFemales[self.expandedDegSeqFemales!=vertexInd]
            
            if self.graph.vlist.V[vertexInd, HIVVertices.orientationIndex]==HIVVertices.bi:
                self.expandedDegSeqBiFemales = self.expandedDegSeqBiFemales[self.expandedDegSeqBiFemales!=vertexInd]    
        
        #Update set of detected neighbours
        for neighbour in self.neighboursList[vertexInd]:
            self.detectedNeighboursList[neighbour] = numpy.append(self.detectedNeighboursList[neighbour], numpy.array([vertexInd])) 

            #Check these are correct
            assert ((numpy.sort(self.detectedNeighboursList[neighbour]) == self.graph.detectedNeighbours(neighbour)).all()), \
                "%s and %s" % (numpy.sort(self.detectedNeighboursList[neighbour]),  self.graph.detectedNeighbours(neighbour))

    def upperContactRates(self, infectedList):
        """
        This is an upper bound on the contact rates not dependant on the time. We
        just return a vector of upper bounds for each infected
        """

        #Note a heterosexual can only have a heterosexual rate but bisexual can have either 
        contactRates = numpy.ones(len(infectedList))*self.contactRate
        #contactRates += (self.graph.vlist.V[infectedList, HIVVertices.orientationIndex])*self.contactRate

        return numpy.sum(contactRates)
        
    def upperDetectionRates(self, infectedList, n, seed=21):
        """
        An upper bound on the detection rates indepedant of time. This is just the
        random detection rate plus the ctRate per person for each detected neighbour. 
        """
        detectionRates = self.randomDetectionRates(infectedList, n, seed=21)

        for i, j in enumerate(infectedList):
            detectionRates[i] += self.detectedNeighboursList[j].shape[0]*self.ctRatePerPerson

        return numpy.sum(detectionRates)

    #@profile 
    def contactRates(self, infectedList, contactList, t):
        """
        Work out contact rates between all infected and all other individuals. The
        set of infected is given in infectedList, and the set of contacts is given
        in contactList. Here we compute rates between an infected and all others
        and then restrict to the people given in contactList.
        """
        if len(infectedList) == 0:
            return numpy.array([]), numpy.array([])

        infectedV = self.graph.vlist.V[infectedList, :]
        
        maleInfectInds = infectedV[:, HIVVertices.genderIndex]==HIVVertices.male
        femaleInfectInds = numpy.logical_not(maleInfectInds)
        biInfectInds = infectedV[:, HIVVertices.orientationIndex]==HIVVertices.bi
        maleBiInfectInds = numpy.logical_and(maleInfectInds, biInfectInds)
        femaleBiInfectInds = numpy.logical_and(femaleInfectInds, biInfectInds)

        #possibleContacts has as its first column the previous contacts.
        #A set of new contacts based on the degree sequence
        #These contacts can be any one of the other contacts that are non-removed
        numPossibleContacts = 2
        possibleContacts = numpy.zeros((len(infectedList), numPossibleContacts), numpy.int)
        possibleContactWeights = numpy.zeros((len(infectedList), numPossibleContacts))
        
        possibleContacts[:, 0] = self.previousContact[infectedList]

        #Note that we may get duplicates for possible contacts since we don't check for it
        edsInds = numpy.random.randint(0, self.expandedDegSeqFemales.shape[0], maleInfectInds.sum())
        contactInds = self.expandedDegSeqFemales[edsInds]
        possibleContacts[maleInfectInds, 1] = contactInds

        edsInds = numpy.random.randint(0, self.expandedDegSeqMales.shape[0], femaleInfectInds.sum())
        contactInds = self.expandedDegSeqMales[edsInds]
        possibleContacts[femaleInfectInds, 1] = contactInds

        if self.expandedDegSeqBiMales.shape[0] != 0:
            edsInds = numpy.random.randint(0, self.expandedDegSeqBiMales.shape[0], maleBiInfectInds.sum())
            contactInds = self.expandedDegSeqBiMales[edsInds]
            possibleContacts[maleBiInfectInds, 1] = contactInds

        if self.expandedDegSeqBiFemales.shape[0] != 0:
            edsInds = numpy.random.randint(0, self.expandedDegSeqBiFemales.shape[0], femaleBiInfectInds.sum())
            contactInds = self.expandedDegSeqBiFemales[edsInds]
            possibleContacts[femaleBiInfectInds, 1] = contactInds

        #Choose randomly between the last contact (if any) and current one
        hadLastContact = numpy.array(self.previousContact[infectedList]!=-1, numpy.int)
        possibleContactWeights[:, 0] = self.alpha*hadLastContact
        possibleContactWeights[:, 1] = 1-self.alpha + self.alpha*(1-hadLastContact)
        

        assert (possibleContactWeights >= numpy.zeros((len(infectedList), numPossibleContacts))).all()
        contactInds = Util.random2Choice(possibleContactWeights).ravel()

        contactRateInds = possibleContacts[(numpy.arange(possibleContacts.shape[0]), contactInds)]
        contactRates = numpy.ones(len(infectedList))*self.contactRate
        
        #print(contactRateInds)
        
        return contactRateInds, contactRates

    """
    Compute the infection probability between an infected and susceptible.
    """
    def infectionProbability(self, vertexInd1, vertexInd2, t):
        """
        This returns the infection probability of an infected person vertexInd1
        and a non-removed vertexInd2. 
        """
        vertex1 = self.graph.vlist.V[vertexInd1, :]
        vertex2 = self.graph.vlist.V[vertexInd2, :]

        if vertex1[HIVVertices.stateIndex]!=HIVVertices.infected or vertex2[HIVVertices.stateIndex]!=HIVVertices.susceptible:
            return 0.0

        if vertex1[HIVVertices.genderIndex] != vertex2[HIVVertices.genderIndex]:
            return self.infectProb
        elif vertex1[HIVVertices.genderIndex] == HIVVertices.male and vertex1[HIVVertices.orientationIndex]==HIVVertices.bi and vertex2[HIVVertices.genderIndex] == HIVVertices.male and vertex2[HIVVertices.orientationIndex]==HIVVertices.bi:
            return self.infectProb
        else:
            #Corresponds to 2 bisexual women 
            return 0.0 

    def randomDetectionRates(self, infectedList, n, seed=21):
        """
        Compute the detection rate of an infected which depends on the entire population.
        In this case it's randDetectRate * |I_t| / n since more infectives means the detection
        rate is higher. The value of n is | I \cup S |.  
        """
        state = numpy.random.get_state()
        numpy.random.seed(seed)
        detectionRates = numpy.ones(len(infectedList)) * self.randDetectRate*len(infectedList)/float(n)
        numpy.random.set_state(state)
        return detectionRates

    def contactTracingRates(self, infectedList, removedSet, t, seed=21):
        """
        Compute the contact tracing detection rate of a list of infected individuals.
        """
        assert set(infectedList).intersection(removedSet) == set([])

        ctRates = numpy.zeros(len(infectedList))
        #A neigbour that was detected at ctStartDate or earlier can result in detection
        ctStartDate = t - self.ctStartTime
        #A detected neigbour that happened later than cdEndDate can result in detection
        cdEndDate = t - self.ctEndTime

        infectedArray = numpy.array(infectedList)
        infectedArrInds = numpy.argsort(infectedArray)
        infectedArray = infectedArray[infectedArrInds]

        removeIndices = numpy.array(list(removedSet), numpy.int)
        underCT = numpy.zeros(self.graph.size, numpy.bool)
        underCT[removeIndices] = numpy.logical_and(self.graph.vlist.V[removeIndices, HIVVertices.detectionTimeIndex] >= cdEndDate, self.graph.vlist.V[removeIndices, HIVVertices.detectionTimeIndex] <= ctStartDate)

        if len(infectedList) < len(removedSet):
            #Possibly store set of detected neighbours
            for i in range(len(infectedList)):
                vertexInd = infectedList[i]
                detectedNeighbours = self.detectedNeighboursList[vertexInd]

                for ind in detectedNeighbours:
                    if underCT[ind]:
                        ctRates[i] = self.ctRatePerPerson
                #This is slower for some reason
                #ctRates[i] = numpy.sum(underCT[neighbours]) * self.ctRatePerPerson
        else:
            for vertexInd in removedSet:
                if underCT[vertexInd]:
                    neighbours = self.neighboursList[vertexInd]

                    for ind in neighbours:
                        if self.graph.vlist.V[ind, HIVVertices.stateIndex] == HIVVertices.infected:
                            i = numpy.searchsorted(infectedArray, ind)
                            ctRates[infectedArrInds[i]] = self.ctRatePerPerson

        assert (ctRates >= numpy.zeros(len(infectedList))).all()

        return ctRates