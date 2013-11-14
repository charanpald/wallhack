import numpy 
import unittest
import logging
import pywt
import pandas 
from apgl.util.PathDefaults import PathDefaults
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils 
import numpy.testing as nptst 

class  MetabolomicsUtilsTestCase(unittest.TestCase):
    def setUp(self): 
        numpy.set_printoptions(threshold=3000)    
    
    def testLoadData(self): 
        metaUtils = MetabolomicsUtils() 
        
        X, X2, (Xopls1, Xopls2, Xopls3), YList, ages = metaUtils.loadData()
        
        #Looks okay 
        #print(X.shape, X2.shape, Xopls1.shape, Xopls2.shape, Xopls3.shape)
        #print(ages)
        #print(YList)
    
    def testCreateIndicatorLabels(self):
        
        metaUtils = MetabolomicsUtils()
        X, X2, (Xopls1, Xopls2, Xopls3), YList, ages = metaUtils.loadData()
        
        Y1, inds1 = YList[0]
        Y2, inds2 = YList[1]
        Y3, inds3 = YList[2]

        YICortisolInds, YTestoInds, YIgf1Inds = metaUtils.createIndicatorLabels(YList)

        s = YICortisolInds[0] + YICortisolInds[1] + YICortisolInds[2]
        nptst.assert_array_equal(s[inds1], numpy.ones(inds1.sum()))

        s = YTestoInds[0] + YTestoInds[1] + YTestoInds[2]
        nptst.assert_array_equal(s[inds2], numpy.ones(inds2.sum()))

        s = YIgf1Inds[0] + YIgf1Inds[1] + YIgf1Inds[2]
        nptst.assert_array_equal(s[inds3], numpy.ones(inds3.sum()))

        #Now compare to those labels in the file
        dataDir = PathDefaults.getDataDir() +  "metabolomic/"
        fileName = dataDir + "data.RMN.total.6.txt"
        data = pandas.read_csv(fileName, delimiter=",") 

        labelNames = [] 
        labelNames.extend(["Ind.Cortisol.1", "Ind.Cortisol.2", "Ind.Cortisol.3"])
        labelNames.extend(["Ind.Testo.1", "Ind.Testo.2", "Ind.Testo.3"])
        labelNames.extend(["Ind.IGF1.1", "Ind.IGF1.2", "Ind.IGF1.3"])
        
        Y = numpy.array(data[labelNames[0]])
        YICortisolInds = numpy.array(YICortisolInds).T
        print(YICortisolInds)
        nptst.assert_almost_equal(YICortisolInds[0][inds1], Y[inds1])
        
        Y = numpy.array(data[labelNames[1]], numpy.int)
        nptst.assert_almost_equal(YICortisolInds[1][inds1], Y[inds1])

        Y = numpy.array(data[labelNames[2]], numpy.int)
        nptst.assert_almost_equal(YICortisolInds[2][inds1], Y[inds1])

        Y = numpy.array(data[labelNames[0]], numpy.int)[inds1]
        nptst.assert_almost_equal(YIgf1Inds[0][inds1], Y)
        Y = numpy.array(data[labelNames[1]], numpy.int)[inds1]
        nptst.assert_almost_equal(YIgf1Inds[1][inds1], Y)
        Y = numpy.array(data[labelNames[2]], numpy.int)[inds1]
        nptst.assert_almost_equal(YIgf1Inds[2][inds1], Y)

        Y = numpy.array(data[labelNames[0]], numpy.int)[inds1]
        nptst.assert_almost_equal(YIgf1Inds[0][inds1], Y)
        Y = numpy.array(data[labelNames[1]], numpy.int)[inds1]
        nptst.assert_almost_equal(YIgf1Inds[1][inds1], Y)
        Y = numpy.array(data[labelNames[2]], numpy.int)[inds1]
        nptst.assert_almost_equal(YIgf1Inds[2][inds1], Y)
        
    @unittest.skip("")
    def testGetWaveletFeaturesTest(self):
        #See if we can reproduce the data from the wavelet 

        X, X2, Xs, Xopls, YList, df = MetabolomicsUtils.loadData()

        waveletStr = 'db4'
        mode = "zpd"
        level = 10
        C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
        X0 = pywt.waverec(C, waveletStr, mode)
        tol = 10**-6
        self.assertTrue(numpy.linalg.norm(X0 - X[0, :]) < tol)

        def reconstructSignal(X, Xw, waveletStr, level, mode, C):
            Xrecstr = numpy.zeros(X.shape)
            
            for i in range(Xw.shape[0]):
                C2 = []

                colIndex = 0
                for j in range(len(list(C))):
                    C2.append(Xw[i, colIndex:colIndex+len(C[j])])
                    colIndex += len(C[j])

                Xrecstr[i, :] = pywt.waverec(tuple(C2), waveletStr, mode)

            return Xrecstr

        #Now do the same for the whole X
        C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
        Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
        Xrecstr = reconstructSignal(X, Xw, waveletStr, level, mode, C)
        self.assertTrue(numpy.linalg.norm(X - Xrecstr) < tol)

        waveletStr = 'db8'
        C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
        Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
        Xrecstr = reconstructSignal(X, Xw, waveletStr, level, mode, C)
        self.assertTrue(numpy.linalg.norm(X - Xrecstr) < tol)

        waveletStr = 'haar'
        C = pywt.wavedec(X[0, :], waveletStr, level=level, mode=mode)
        Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
        Xrecstr = reconstructSignal(X, Xw, waveletStr, level, mode, C)
        self.assertTrue(numpy.linalg.norm(X - Xrecstr) < tol)
    
    @unittest.skip("")    
    def testScoreLabel(self):#
        numExamples = 10 
        Y = numpy.random.rand(numExamples)

        bounds = numpy.array([0, 0.2, 0.8, 1.0])

        YScores = MetabolomicsUtils.scoreLabels(Y, bounds)

        inds1 = numpy.argsort(Y)
        inds2 = numpy.argsort(YScores[:, 0])
        inds3 = numpy.argsort(YScores[:, -1])

        inds4 = numpy.argsort(numpy.abs(Y - 0.5))
        inds5 = numpy.argsort(YScores[:, 1])

        self.assertTrue((inds1 == inds3).all())
        self.assertTrue((inds1 == numpy.flipud(inds2)).all())
        self.assertTrue((inds4 == numpy.flipud(inds5)).all())

        #Test we don't get problems when Y has the same values
        Y = numpy.ones(numExamples)
        YScores = MetabolomicsUtils.scoreLabels(Y, bounds)

        self.assertTrue((YScores == numpy.ones((Y.shape[0], 3))).all())

    @unittest.skip("")
    def testReconstructSignal(self):
        numExamples = 100 
        numFeatures = 16 
        X = numpy.random.rand(numExamples, numFeatures)

        level = 10 
        mode = "cpd"
        waveletStr = "db4"
        C = pywt.wavedec(X[0, :], waveletStr, mode, level=10)

        Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
        X2 = MetabolomicsUtils.reconstructSignal(X, Xw, waveletStr, mode, C)

        tol = 10**-6 
        self.assertTrue(numpy.linalg.norm(X - X2) < tol)

    @unittest.skip("")
    def testFilterWavelet(self):
        numExamples = 100
        numFeatures = 16
        X = numpy.random.rand(numExamples, numFeatures)

        level = 10
        mode = "cpd"
        waveletStr = "db4"
        C = pywt.wavedec(X[0, :], waveletStr, mode, level=10)

        Xw = MetabolomicsUtils.getWaveletFeatures(X, waveletStr, level, mode)
        
        N = 10
        Xw2, inds = MetabolomicsUtils.filterWavelet(Xw, N)

        tol = 10**-6 
        self.assertEquals(inds.shape[0], N)
        self.assertTrue(numpy.linalg.norm( Xw[:, inds] - Xw2[:, inds] ) < tol)

        zeroInds = numpy.setdiff1d(numpy.arange(Xw.shape[1]), inds)
        self.assertTrue(numpy.linalg.norm(Xw2[:, zeroInds]) < tol)

if __name__ == '__main__':
    unittest.main()

