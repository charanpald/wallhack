import argparse
import numpy
import sys
import logging
import os
import datetime
import multiprocessing 
from sandbox.util.PathDefaults import PathDefaults
from wallhack.metabolomics.MetabolomicsExpHelper import MetabolomicsExpHelper
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from socket import gethostname
from sklearn.decomposition import PCA

"""
Run a variety of bipartite ranking on the metabolomics data 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.debug("Running from machine " + str(gethostname()))
numpy.random.seed(21)

os.system('taskset -p 0xffffffff %d' % os.getpid())

dataDir = PathDefaults.getDataDir() +  "metabolomic/"
metaUtils = MetabolomicsUtils() 
X, XStd, X2, (XoplsCortisol, XoplsTesto, XoplsIgf1), YCortisol, YTesto, YIgf1, ages = metaUtils.loadData()

#We model 99.1% of the spectrum with 100 eigenvectors 
pca = PCA(n_components=100)
XPca = pca.fit_transform(X)

mode = "cpd"
level = 10
XwDb4 = MetabolomicsUtils.getWaveletFeatures(X, 'db4', level, mode)
XwDb8 = MetabolomicsUtils.getWaveletFeatures(X, 'db8', level, mode)
XwHaar = MetabolomicsUtils.getWaveletFeatures(X, 'haar', level, mode)

dataDict = {}
dataDict["raw"] = X
dataDict["pca"] = XPca
#dataDict["log"] = X2
dataDict["Db4"] = XwDb4
dataDict["Db8"] = XwDb8
dataDict["Haar"] = XwHaar 


numpy.random.seed(datetime.datetime.now().microsecond)

parser = argparse.ArgumentParser(description='Run the metabolomics experiments')
parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count())
parser.add_argument("--runAll", action="store_true", default=False)
parser.add_argument("--runCartTR", action="store_true", default=False)
parser.add_argument("--runRbfSvmTR", action="store_true", default=False)
parser.add_argument("--runL1SvmTR", action="store_true", default=False)
parser.add_argument("--runCartTRF", action="store_true", default=False)
parser.add_argument("--runRbfSvmTRF", action="store_true", default=False)
parser.add_argument("--runL1SvmTRF", action="store_true", default=False)
parser.add_argument("--runRankSVM", action="store_true", default=False)
parser.add_argument("--runRankBoost", action="store_true", default=False)

parser.add_argument("--hormoneInd", type=int, default=3, help='Choose which hormone to work on: cortisol (0), testosterone (1), IGF1 (2), all(3)')
args = parser.parse_args()

if args.hormoneInd == 0: 
    runCortisol = True 
    runTestosterone = False 
    runIGF1 = False 
elif args.hormoneInd == 1:
    runCortisol = False 
    runTestosterone = True 
    runIGF1 = False 
elif args.hormoneInd == 2:
    runCortisol = False 
    runTestosterone = False 
    runIGF1 = True 
elif args.hormoneInd == 3:
    runCortisol = True 
    runTestosterone = True 
    runIGF1 = True 
else:
    raise ValueError("Invalid hormone index: " + str(args.hormoneInd))

helper = MetabolomicsExpHelper(dataDict, YCortisol, YTesto, YIgf1, ages, args.processes, runCortisol, runTestosterone, runIGF1)

if args.runAll: 
    #helper.runCartTreeRank = True
    #helper.runRbfSvmTreeRank = True
    #helper.runL1SvmTreeRank = True
    helper.runCartTreeRankForest = True
    helper.runRbfSvmTreeRankForest = True
    helper.runL1SvmTreeRankForest = True
    #helper.runRankSVM = True
    helper.runRankBoost = True
else: 
    helper.runCartTreeRank = args.runCartTR
    helper.runRbfSvmTreeRank = args.runRbfSvmTR
    helper.runL1SvmTreeRank = args.runL1SvmTR
    helper.runCartTreeRankForest = args.runCartTRF
    helper.runRbfSvmTreeRankForest = args.runRbfSvmTRF
    helper.runL1SvmTreeRankForest = args.runL1SvmTRF
    helper.runRankSVM = args.runRankSVM
    helper.runRankBoost = args.runRankBoost

helper.run()
