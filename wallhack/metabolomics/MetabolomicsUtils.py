
import pywt
import numpy
import pandas 
from apgl.util.PathDefaults import PathDefaults
from sandbox.data.Standardiser import Standardiser

class MetabolomicsUtils(object):
    def __init__(self):
        self.labelNames = ["Cortisol.val", "Testosterone.val", "IGF1.val"]
        self.dataDir = PathDefaults.getDataDir() +  "metabolomic/"
        self.boundsDict = {}
        self.boundsDict["Cortisol"] = numpy.array([0, 89, 225, 573])
        self.boundsDict["Testosterone"] = numpy.array([0, 3, 9, 13])
        self.boundsDict["IGF1"] = numpy.array([0, 200, 441, 782])
        
    def getLabelNames(self):
        return self.labelNames

    def getBounds(self):
        """
        Return the bounds used to define the indicator variables
        """
        return self.boundsList 

    def loadData(self):
        """
        Return the raw spectra and the MDS transformed data as well as the DataFrame
        for the MDS data. 
        """
        fileName = self.dataDir + "data.RMN.total.6.txt"
        data = pandas.read_csv(fileName, delimiter=",")        
        
        maxNMRIndex = 951
        X = data.iloc[:, 1:maxNMRIndex].values
        XStd = Standardiser().standardiseArray(X)

        #Load age and normalise (missing values are assigned the mean) 
        ages = numpy.array(data["Age"])
        meanAge = numpy.mean(data["Age"])
        ages[numpy.isnan(ages)] = meanAge
        ages = Standardiser().standardiseArray(ages)
        
        #Load labels 
        YCortisol = data["Cortisol.val"]
        YTesto = data["Testosterone.val"]
        YIgf1 = data["IGF1.val"]

        fileName = self.dataDir + "data.sportsmen.log.AP.1.txt"
        maxNMRIndex = 419
        data = pandas.read_csv(fileName, delimiter=",")  
        X2 = data.iloc[:, 1:maxNMRIndex].values
        X2 = Standardiser().standardiseArray(X2)

        #Load the OPLS corrected files
        fileName = self.dataDir + "cort.log.OSC.1.txt"
        minNMRIndex = 20
        maxNMRIndex = 439
        data = pandas.read_csv(fileName, delimiter=",")  
        XoplsCortisol = data.iloc[:, minNMRIndex:maxNMRIndex].values
        XoplsCortisol = Standardiser().standardiseArray(XoplsCortisol)

        fileName = self.dataDir + "testo.log.OSC.1.txt"
        minNMRIndex = 22
        maxNMRIndex = 441
        data = pandas.read_csv(fileName, delimiter=",")  
        XoplsTesto = data.iloc[:, minNMRIndex:maxNMRIndex].values
        XoplsTesto = Standardiser().standardiseArray(XoplsTesto)
        
        fileName = self.dataDir + "IGF1.log.OSC.1.txt"
        minNMRIndex = 22
        maxNMRIndex = 441
        data = pandas.read_csv(fileName, delimiter=",")  
        XoplsIgf1 = data.iloc[:, minNMRIndex:maxNMRIndex].values
        XoplsIgf1 = Standardiser().standardiseArray(XoplsIgf1)
        
        return X, XStd, X2, (XoplsCortisol, XoplsTesto, XoplsIgf1), YCortisol, YTesto, YIgf1, ages

    @staticmethod 
    def getWaveletFeatures(X, waveletStr, level, mode):
        """
        Give a matrix of signals in the rows X, compute a wavelet given by waveletStr
        with given level and extension mode.
        """
        C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
        numFeatures = 0

        for c in list(C):
            numFeatures += len(c)

        Xw = numpy.zeros((X.shape[0], numFeatures))

        #Compute wavelet features
        for i in range(X.shape[0]):
            C = pywt.wavedec(X[i, :], waveletStr, level=level, mode="zpd")

            colInd = 0
            for j in range(len(C)):
                Xw[i, colInd:colInd+C[j].shape[0]] = C[j]
                colInd += C[j].shape[0]

        return Xw

    def createIndicatorLabel(self, Y, bounds):
        """
        Given a set of concentrations and bounds, create a set of indicator label
        """
        YInds = []
        nonMissingInds = numpy.logical_not(numpy.isnan(Y))

        for i in range(bounds.shape[0]-1):
            YInd = numpy.zeros(Y.shape)
            YInd[nonMissingInds] = numpy.array(numpy.logical_and(bounds[i] < Y[nonMissingInds], Y[nonMissingInds] <= bounds[i+1]), numpy.int) 
            YInds.append(YInd) 
        
        YInds = numpy.array(YInds).T        
        
        return YInds

    @staticmethod 
    def scoreLabels(Y, bounds):
        """
        Take a set of predicted labels Y and score them within a vector of bounds.
        """

        numIndicators = bounds.shape[0]-1
        YScores = numpy.zeros((Y.shape[0], numIndicators))

        YScores[:, 0] = Y - bounds[0]

        for i in range(1, bounds.shape[0]-1):
            YScores[:, i] = numpy.abs((bounds[i+1]+bounds[i])/2 - Y)

        YScores[:, -1] = bounds[-1]- Y
        YScores = (YScores-numpy.min(YScores, 0))
        maxVals = numpy.max(YScores, 0) + numpy.array(numpy.max(YScores, 0)==0, numpy.float)
        YScores = 1 - YScores/maxVals

        return YScores

    @staticmethod
    def reconstructSignal(X, Xw, waveletStr, mode, C):
        Xrecstr = numpy.zeros(X.shape)

        for i in range(Xw.shape[0]):
            C2 = []

            colIndex = 0
            for j in range(len(list(C))):
                C2.append(Xw[i, colIndex:colIndex+len(C[j])])
                colIndex += len(C[j])

            Xrecstr[i, :] = pywt.waverec(tuple(C2), waveletStr, mode)

        return Xrecstr

    @staticmethod
    def filterWavelet(Xw, N):
        """
        Pick the N largest features. 
        """
        inds = numpy.flipud(numpy.argsort(numpy.sum(Xw**2, 0)))[0:N]
        zeroInds = numpy.setdiff1d(numpy.arange(Xw.shape[1]), inds)

        Xw2 = Xw.copy()
        Xw2[:, zeroInds] = 0

        return Xw2, inds