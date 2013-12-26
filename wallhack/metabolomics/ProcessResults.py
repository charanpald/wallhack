import sys
import numpy 
import logging
import datetime 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Latex import Latex 
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from wallhack.metabolomics.MetabolomicsExpHelper import MetabolomicsExpHelper

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
resultsDir = PathDefaults.getOutputDir() + "metabolomics/"
figureDir = resultsDir + "Figures/"

metaUtils = MetabolomicsUtils() 
X, XStd, X2, (XoplsCortisol, XoplsTesto, XoplsIgf1), YCortisol, YTesto, YIgf1, ages = metaUtils.loadData()

dataDict = {}
numpy.random.seed(datetime.datetime.now().microsecond)
helper = MetabolomicsExpHelper(dataDict, YCortisol, YTesto, YIgf1, ages)

dataNames =[] 
dataNames.extend(["raw", "pca", "log", "Db4", "Db8", "Haar"])
#algorithms = ["CartTreeRank", "CartTreeRankForest", "L1SvmTreeRank", "L1SvmTreeRankForest", "RbfSvmTreeRank", "RbfSvmTreeRankForest", "RankBoost", "RankSVM"]
algorithms = ["CartTreeRankForest", "L1SvmTreeRankForest", "RbfSvmTreeRankForest", "RankBoost", "RankSVM"]
algorithmsAbbr = ["Cart-TRF", "L1-TRF", "RBF-TRF", "RB", "RSVM"]

hormoneNameIndicators = [] 
for i, (hormoneName, hormoneConc) in enumerate(helper.hormoneDict.items()):
    hormoneIndicators = metaUtils.createIndicatorLabel(hormoneConc, metaUtils.boundsDict[hormoneName])
    for j in range(hormoneIndicators.shape[1]):
        hormoneNameIndicators.append(hormoneName + "-" +str(j))

numIndicators = 3 
testAucsMean = numpy.zeros((len(hormoneNameIndicators), len(dataNames), len(algorithms)))
testAucsStd = numpy.zeros((len(hormoneNameIndicators), len(dataNames), len(algorithms)))

for i, hormoneNameIndicator in enumerate(hormoneNameIndicators):
    for j, dataName in enumerate(dataNames):
        for k, alg in enumerate(algorithms): 
            fileName = resultsDir + alg + "-" + hormoneNameIndicator + "-" + dataName + ".npy"
            
            try: 
                errors = numpy.load(fileName)
                testAucsMean[i, j, k] = numpy.mean(errors)
                testAucsStd[i, j, k] = numpy.std(errors)
                logging.debug("Read file: " + fileName)
            except: 
                logging.debug("File not found : " + str(fileName))
    
for i, dataName in enumerate(dataNames): 
    print("-"*10 + dataName + "-"*10)

    algorithms = [x.ljust(20) for x in algorithmsAbbr]
    table = Latex.array2DToRows(testAucsMean[:, i, :].T, testAucsStd[:, i, :].T, precision=2)
    print(Latex.listToRow(hormoneNameIndicators))
    print(Latex.addRowNames(algorithms, table))