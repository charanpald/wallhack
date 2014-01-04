import sys
import numpy 
import logging
import datetime
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt  
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Latex import Latex 
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from wallhack.metabolomics.MetabolomicsExpHelper import MetabolomicsExpHelper

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=3)
resultsDir = PathDefaults.getOutputDir() + "metabolomics/"
figureDir = resultsDir + "Figures/"

metaUtils = MetabolomicsUtils() 
X, XStd, X2, (XoplsCortisol, XoplsTesto, XoplsIgf1), YCortisol, YTesto, YIgf1, ages = metaUtils.loadData()

dataDict = {}
numpy.random.seed(datetime.datetime.now().microsecond)
helper = MetabolomicsExpHelper(dataDict, YCortisol, YTesto, YIgf1, ages)

dataNames =[] 
dataNames.extend(["raw", "pca", "Db4", "Db8", "Haar"])
#algorithms = ["CartTreeRank", "CartTreeRankForest", "L1SvmTreeRank", "L1SvmTreeRankForest", "RbfSvmTreeRank", "RbfSvmTreeRankForest", "RankBoost", "RankSVM"]
algorithms = ["CartTreeRankForest", "L1SvmTreeRankForest", "RbfSvmTreeRankForest", "RankBoost", "RankSVM"]
algorithmsAbbr = ["CART-TRF", "L1-TRF", "RBF-TRF", "RB", "RSVM"]

hormoneNameIndicators = [] 
for i, (hormoneName, hormoneConc) in enumerate(helper.hormoneDict.items()):
    hormoneIndicators = metaUtils.createIndicatorLabel(hormoneConc, metaUtils.boundsDict[hormoneName])
    for j in range(hormoneIndicators.shape[1]):
        hormoneNameIndicators.append(hormoneName + "-" +str(j))

numIndicators = 3 
testAucsMean = numpy.zeros((len(hormoneNameIndicators), len(dataNames), len(algorithms)))
testAucsStd = numpy.zeros((len(hormoneNameIndicators), len(dataNames), len(algorithms)))

numMissingFiles = 0

for i, hormoneNameIndicator in enumerate(hormoneNameIndicators):
    for j, dataName in enumerate(dataNames):
        for k, alg in enumerate(algorithms): 
            fileName = resultsDir + alg + "-" + hormoneNameIndicator + "-" + dataName + ".npy"
            
            try: 
                errors = numpy.load(fileName)
                testAucsMean[i, j, k] = numpy.mean(errors)
                testAucsStd[i, j, k] = numpy.std(errors)
                #logging.debug("Read file: " + fileName)
            except: 
                logging.debug("File not found : " + str(fileName))
                numMissingFiles += 1 
                
logging.debug("Number of missing files: " + str(numMissingFiles))
    
for i, dataName in enumerate(dataNames): 
    print("-"*10 + dataName + "-"*10)

    algorithms = [x.ljust(20) for x in algorithmsAbbr]
    currentTestAucsMean = testAucsMean[:, i, :].T
    maxAUCs = numpy.zeros(currentTestAucsMean.shape, numpy.bool)
    maxAUCs[numpy.argmax(currentTestAucsMean, 0), numpy.arange(currentTestAucsMean.shape[1])] = 1
    table = Latex.array2DToRows(testAucsMean[:, i, :].T, testAucsStd[:, i, :].T, precision=2, bold=maxAUCs)
    print(Latex.listToRow(hormoneNameIndicators))
    print(Latex.addRowNames(algorithms, table))
    

#Now looks at the features for the raw spectra 
algorithm = "L1SvmTreeRankForest" 
dataName = "raw"
numMissingFiles = 0 
numFeatures = 100

numIndicators = 6 
featureInds = numpy.zeros((numFeatures, numIndicators))

for i, (hormoneName, hormoneConc) in enumerate(helper.hormoneDict.items()):
    try: 
        fileName = resultsDir + "Weights" + algorithm + "-" + hormoneName + "-0" + "-" + dataName + ".npy"
        weights = numpy.load(fileName)
        weights = weights/numpy.linalg.norm(weights)
        
        inds0 = numpy.flipud(numpy.argsort(weights))[0:numFeatures]

        #print(inds0 == 950)
        featureInds[:, 2*i] = inds0
    except IOError: 
        logging.debug("File not found : " + str(fileName))
        numMissingFiles += 1 
        
    try: 
        fileName = resultsDir + "Weights" + algorithm + "-" + hormoneName + "-2" + "-" + dataName + ".npy"
        weights = numpy.load(fileName)
        weights = weights/numpy.linalg.norm(weights)
        
        inds2 = numpy.flipud(numpy.argsort(weights))[0:numFeatures]
        #print(inds2 == 950)
        featureInds[:, 2*i+1] = inds2
    except IOError: 
        logging.debug("File not found : " + str(fileName))
        numMissingFiles += 1 
        
    print(numpy.intersect1d(inds0, inds2).shape)
        
print(Latex.array2DToRows(featureInds, precision=0))