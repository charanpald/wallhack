import numpy 
import unittest
import logging
import pywt
import pandas 
from sandbox.util.PathDefaults import PathDefaults
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils 
import numpy.testing as nptst 

class  MetabolomicsUtilsTestCase(unittest.TestCase):
    def setUp(self): 
        numpy.set_printoptions(threshold=3000)    
    
    def testLoadData(self): 
        metaUtils = MetabolomicsUtils() 
        
        X, XStd, X2, (XoplsCortisol, XoplsTesto, XoplsIgf1), YCortisol, YTesto, YIgf1, ages = metaUtils.loadData()
        
        #Looks okay 
        #print(X.shape, X2.shape, Xopls1.shape, Xopls2.shape, Xopls3.shape)
        #print(ages)
        #print(YList)
    
    def testCreateIndicatorLabels(self):
        metaUtils = MetabolomicsUtils()
        X, XStd, X2, (XoplsCortisol, XoplsTesto, XoplsIgf1), YCortisol, YTesto, YIgf1, ages = metaUtils.loadData()
        
        YCortisol = YCortisol[numpy.logical_not(numpy.isnan(YCortisol))]
        YCortisolIndicators = metaUtils.createIndicatorLabel(YCortisol, metaUtils.boundsDict["Cortisol"])
        
        YTesto = YTesto[numpy.logical_not(numpy.isnan(YTesto))]
        YTestoIndicators = metaUtils.createIndicatorLabel(YTesto, metaUtils.boundsDict["Testosterone"])
        
        YIgf1 = YIgf1[numpy.logical_not(numpy.isnan(YIgf1))]
        YIgf1Indicators = metaUtils.createIndicatorLabel(YIgf1, metaUtils.boundsDict["IGF1"])

        s = numpy.sum(YCortisolIndicators, 1)
        nptst.assert_array_equal(s, numpy.ones(s.shape[0]))

        s = numpy.sum(YTestoIndicators, 1)
        nptst.assert_array_equal(s, numpy.ones(s.shape[0]))

        s = numpy.sum(YIgf1Indicators, 1)
        nptst.assert_array_equal(s, numpy.ones(s.shape[0]))

        #Now compare to those labels in the file
        X, X2, (XoplsCortisol, XoplsTesto, XoplsIgf1), YCortisol, YTesto, YIgf1, ages = metaUtils.loadData()
        dataDir = PathDefaults.getDataDir() +  "metabolomic/"
        fileName = dataDir + "data.RMN.total.6.txt"
        data = pandas.read_csv(fileName, delimiter=",") 

        YCortisolIndicators = metaUtils.createIndicatorLabel(YCortisol, metaUtils.boundsDict["Cortisol"])
        YCortisolIndicators2 = numpy.array(data[["Ind.Cortisol.1", "Ind.Cortisol.2", "Ind.Cortisol.3"]])
        
        for i in range(YCortisolIndicators.shape[0]): 
            if not numpy.isnan(YCortisol[i]) and not numpy.isnan(YCortisolIndicators2[i, :]).any(): 
                #nptst.assert_almost_equal(YCortisolIndicators2[i, :], YCortisolIndicators[i, :])
                pass 
        
        YTestoIndicators = metaUtils.createIndicatorLabel(YTesto, metaUtils.boundsDict["Testosterone"])
        YTestoIndicators2 = numpy.array(data[["Ind.Testo.1", "Ind.Testo.2", "Ind.Testo.3"]])
        
        for i in range(YTestoIndicators.shape[0]): 
            if not numpy.isnan(YTesto[i]) and not numpy.isnan(YTestoIndicators2[i, :]).any(): 
                #print(i, YTesto[i])
                nptst.assert_almost_equal(YTestoIndicators2[i, :], YTestoIndicators[i, :])
                
        YIgf1Indicators = metaUtils.createIndicatorLabel(YIgf1, metaUtils.boundsDict["IGF1"])
        YIgf1Indicators2 = numpy.array(data[["Ind.IGF1.1", "Ind.IGF1.2", "Ind.IGF1.3"]])
        
        for i in range(YIgf1Indicators.shape[0]): 
            if not numpy.isnan(YIgf1[i]) and not numpy.isnan(YIgf1Indicators2[i, :]).any(): 
                #print(i, YIgf1[i])
                #nptst.assert_almost_equal(YIgf1Indicators2[i, :], YIgf1Indicators[i, :])
                pass
        #Note that there are some errors in the indicators labels for Cortisol and IGF1 
        #but we will take concentrations as the base truth 
        
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

