import sys
import numpy 
import logging
import datetime 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Latex import Latex 
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from wallhack.metabolomics.MetabolomicsExpHelper import MetabolomicsExpHelper

logging.basicConfig(stream=sys.stdout, level=logging.WARN)
resultsDir = PathDefaults.getOutputDir() + "metabolomics/"
figureDir = resultsDir + "Figures/"

metaUtils = MetabolomicsUtils() 
X, XStd, X2, (XoplsCortisol, XoplsTesto, XoplsIgf1), YCortisol, YTesto, YIgf1, ages = metaUtils.loadData()

Ns = [25, 50, 75]
dataDict = {}
numpy.random.seed(datetime.datetime.now().microsecond)
helper = MetabolomicsExpHelper(dataDict, YCortisol, YTesto, YIgf1, ages)

dataNames =[] 
for i, N in enumerate(Ns):
    dataNames.append("Db4-" + str(N))
    dataNames.append("Db8-" + str(N))
    dataNames.append("Haar-" + str(N))

dataNames.extend(["raw", "Db4", "Db8", "Haar", "log"])
algorithms = ["CartTreeRank", "CartTreeRankForest", "RbfSvmTreeRank", "RbfSvmTreeRankForest", "RankBoost", "RankSVM"]

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
                print(fileName, numpy.mean(errors))
            except: 
                logging.debug("File not found : " + str(fileName))
    
for i, dataName in enumerate(dataNames): 
    print("-"*10 + dataName + "-"*10)

    algorithms = [x.ljust(20) for x in algorithms]
    table = Latex.array2DToRows(testAucsMean[:, i, :].T, testAucsStd[:, i, :].T)
    print(Latex.listToRow(hormoneNameIndicators))
    print(Latex.addRowNames(algorithms, table))