
import pywt
import numpy
from apgl.util.PathDefaults import PathDefaults
from sandbox.data.Standardiser import Standardiser

class MetabolomicsUtils(object):
    def __init__(self):
        self.labelNames = ["IGF1.val", "Cortisol.val", "Testosterone.val"]

    def getLabelNames(self):
        return self.labelNames

    @staticmethod
    def getBounds():
        """
        Return the bounds used to define the indicator variables
        """
        boundsList = []
        boundsList.append(numpy.array([0, 200, 441, 782]))
        boundsList.append(numpy.array([0, 89, 225, 573]))
        boundsList.append(numpy.array([0, 3, 9, 13]))

        return boundsList 

    def loadData(self):
        """
        Return the raw spectra and the MDS transformed data as well as the DataFrame
        for the MDS data. 
        """
        dataDir = PathDefaults.getDataDir() +  "metabolomic/"
        fileName = dataDir + "data.RMN.total.6.txt"
        maxNMRIndex = 951
        X = numpy.loadtxt(fileName, skiprows=1, usecols=range(1, maxNMRIndex), delimiter=",")

        #Load age and normalise (missing values are assinged the mean) 
        ages = numpy.loadtxt(fileName, skiprows=1, usecols=(955,), delimiter=",")
        meanAge = numpy.mean(ages[numpy.logical_not(numpy.isnan(ages))])
        ages[numpy.isnan(ages)] = meanAge
        ages = Standardiser().standardiseArray(ages)

        standardiser = Standardiser()
        X = standardiser.standardiseArray(X)

        fileName = dataDir + "data.sportsmen.log.AP.1.txt"
        maxNMRIndex = 419
        X2 = numpy.loadtxt(fileName, skiprows=1, usecols=range(1, maxNMRIndex), delimiter=",")

        #Load the OPLS corrected files
        fileName = dataDir + "IGF1.log.OSC.1.txt"
        minNMRIndex = 22
        maxNMRIndex = 441
        Xopls1 = numpy.loadtxt(fileName, skiprows=1, usecols=range(minNMRIndex, maxNMRIndex), delimiter=",")

        fileName = dataDir + "cort.log.OSC.1.txt"
        minNMRIndex = 20
        maxNMRIndex = 439
        Xopls2 = numpy.loadtxt(fileName, skiprows=1, usecols=range(minNMRIndex, maxNMRIndex), delimiter=",")

        fileName = dataDir + "testo.log.OSC.1.txt"
        minNMRIndex = 22
        maxNMRIndex = 441
        Xopls3 = numpy.loadtxt(fileName, skiprows=1, usecols=range(minNMRIndex, maxNMRIndex), delimiter=",")

        #Let's load all the label data here
        labelNames = self.getLabelNames()
        YList = MetabolomicsUtils.createLabelList(df, labelNames)
        
        return X, X2, (Xopls1, Xopls2, Xopls3), YList, ages

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

    @staticmethod
    def createLabelList(df, labelNames):
        """
        Given a dataFrame df and list of labelNames, create a list of the vectors
        of labels (Y, inds) in which Y has missing valued removed and inds is a
        boolean vector of non-missing values 
        """
        baseLib = importr('base')
        YList = []

        for labelName in labelNames:
            Y = df.rx(labelName)
            #Finds non missing values and their indices 
            inds = numpy.logical_not(numpy.array(baseLib.is_na(Y), numpy.bool)).ravel()
            Y = numpy.array(Y).ravel()[inds]
            YList.append((Y, inds))

        return YList

    @staticmethod
    def createIndicatorLabel(Y, bounds):
        """
        Given a set of concentrations and bounds, create a set of indicator label
        """
        YInds = []

        for i in range(bounds.shape[0]-1):
            YInds.append(numpy.array(numpy.logical_and(bounds[i] <= Y, Y < bounds[i+1]), numpy.int ) )
        
        return YInds

    @staticmethod
    def createIndicatorLabels(YList):
        """
        Take a list of concentrations for the hormones and return a list of indicator
        variables. 
        """
        boundsList = MetabolomicsUtils.getBounds()
        YIgf1, inds = YList[0]
        YIgf1Inds = MetabolomicsUtils.createIndicatorLabel(YIgf1, boundsList[0])

        YCortisol, inds = YList[1]
        YICortisolInds = MetabolomicsUtils.createIndicatorLabel(YCortisol, boundsList[1])

        YTesto, inds = YList[2]
        YTestoInds = MetabolomicsUtils.createIndicatorLabel(YTesto, boundsList[2])

        return YIgf1Inds, YICortisolInds, YTestoInds

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