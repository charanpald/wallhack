
"""
Start with a simple toy dataset with time-varying characteristics 
"""

import numpy 
import logging
import scipy.sparse 
from sandbox.util.SparseUtils import SparseUtils 
from sandbox.util.SparseUtilsCython import SparseUtilsCython

class SyntheticDataset1(object): 
    def __init__(self, startM=5000, endM=10000, startN=1000, endN=1500, pnz=0.02, noise=0.05, nonUniform=False): 
        self.startM = startM 
        self.endM = endM 
        self.startN = startN 
        self.endN = endN 
        
        self.pnz = pnz
        self.noise = noise
        self.trainSplit = 1.0/2 
        self.nonUniform = nonUniform 
    
    def generateMatrices(self):
        """
        This function returns a list of 20 train/test matrices for incremental 
        collaborative filtering. Each item in the list is (trainX, testX).
        """    
        numpy.random.seed(21)    
        r = 50 
        
        U, s, V = SparseUtils.generateLowRank((self.endM, self.endN), r, normalise=False)
        
        self.startNumInds = self.pnz*self.startM*self.startN
        self.endNumInds = self.pnz*self.endM*self.endN
        
        if not self.nonUniform: 
            inds = numpy.random.randint(0, self.endM*self.endN-1, self.endNumInds)
        else:
            logging.debug("Using non uniform dataset")
            inds = numpy.array(numpy.random.randn(self.endNumInds)*(self.endM*self.endN-1)/4 +(self.endM*self.endN-1)/2, numpy.int)
            inds = numpy.clip(inds, 0, (self.endM*self.endN-1))
            
        inds = numpy.unique(inds)
        numpy.random.shuffle(inds)
        self.endNumInds = inds.shape[0]
        
        rowInds, colInds = numpy.unravel_index(inds, (self.endM, self.endN))
        rowInds = numpy.array(rowInds, numpy.int32)
        colInds = numpy.array(colInds, numpy.int32)
        vals = SparseUtilsCython.partialReconstructValsPQ(rowInds, colInds, U*s, V)
        vals /= vals.std()
        vals +=  numpy.random.randn(vals.shape[0])*self.noise
        
        
        isTrainInd = numpy.array(numpy.random.rand(inds.shape[0]) <= self.trainSplit, numpy.bool)
        
        assert (self.trainSplit - isTrainInd.sum()/float(isTrainInd.shape[0]))
        
        XMaskTrain = scipy.sparse.csc_matrix((isTrainInd, (rowInds, colInds)), dtype=numpy.bool, shape=(self.endM, self.endN))
        XMaskTest = scipy.sparse.csc_matrix((numpy.logical_not(isTrainInd), (rowInds, colInds)), dtype=numpy.bool, shape=(self.endM, self.endN))

        #In the first phase, the matrices stay the same size but there are more nonzero 
        #entries   
        numMatrices = 10 
        stepList = numpy.linspace(self.startNumInds, self.endNumInds, numMatrices) 
        trainXList = []
        testXList = []    
        
        for i in range(numMatrices):  
            currentVals = vals[0:stepList[i]]
            currentRowInds = rowInds[0:stepList[i]]
            currentColInds = colInds[0:stepList[i]]
            
            X = scipy.sparse.csc_matrix((currentVals, (currentRowInds, currentColInds)), dtype=numpy.float, shape=(self.endM, self.endN))
            #print("pnz=" + str(X.nnz/float(X.shape[0]*X.shape[1])))
            
            trainX = X.multiply(XMaskTrain)[0:self.startM, 0:self.startN]
            trainX.eliminate_zeros()
            trainX.prune() 
            
            testX = X.multiply(XMaskTest)[0:self.startM, 0:self.startN]
            testX.eliminate_zeros()
            testX.prune() 
            
            trainXList.append(trainX)
            testXList.append(testX)
            
        #Now we increase the size of matrix 
        numMatrices = 10 
        mStepList = numpy.linspace(self.startM, self.endM, numMatrices)
        nStepList = numpy.linspace(self.startN, self.endN, numMatrices)
    
        X = scipy.sparse.csc_matrix((vals, (rowInds, colInds)), dtype=numpy.float, shape=(self.endM, self.endN))
    
        for i in range(numMatrices): 
            trainX = X.multiply(XMaskTrain)[0:mStepList[i], :][:, 0:nStepList[i]]
            trainX.eliminate_zeros()
            trainX.prune() 
            
            testX = X.multiply(XMaskTest)[0:mStepList[i], :][:, 0:nStepList[i]]
            testX.eliminate_zeros()
            testX.prune() 
            
            trainXList.append(trainX)
            testXList.append(testX)
                    
        return trainXList, testXList
        
    def getTrainIteratorFunc(self):
        def trainIteratorFunc(): 
            trainXList, testXList = self.generateMatrices()       
            return iter(trainXList)
        
        return trainIteratorFunc
        
    def getTestIteratorFunc(self):
        def testIteratorFunc(): 
            trainXList, testXList = self.generateMatrices()   
            return iter(testXList)
        
        return testIteratorFunc
