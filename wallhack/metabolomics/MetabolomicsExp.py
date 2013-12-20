import argparse
import numpy
import sys
import logging
import os
import datetime
from apgl.util.PathDefaults import PathDefaults
from wallhack.metabolomics.MetabolomicsExpHelper import MetabolomicsExpHelper
from wallhack.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from socket import gethostname

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

mode = "cpd"
level = 10
XwDb4 = MetabolomicsUtils.getWaveletFeatures(X, 'db4', level, mode)
XwDb8 = MetabolomicsUtils.getWaveletFeatures(X, 'db8', level, mode)
XwHaar = MetabolomicsUtils.getWaveletFeatures(X, 'haar', level, mode)

dataDict = {}
dataDict["raw"] = X
dataDict["Db4"] = XwDb4
dataDict["Db8"] = XwDb8
dataDict["Haar"] = XwHaar 
dataDict["log"] = X2

numpy.random.seed(datetime.datetime.now().microsecond)

parser = argparse.ArgumentParser(description='Run the metabolomics experiments')
parser.add_argument("--runCartTR", action="store_true", default=False)
parser.add_argument("--runRbfSvmTR", action="store_true", default=False)
parser.add_argument("--runL1SvmTR", action="store_true", default=False)
parser.add_argument("--runCartTRF", action="store_true", default=False)
parser.add_argument("--runRbfSvmTRF", action="store_true", default=False)
parser.add_argument("--runL1SvmTRF", action="store_true", default=False)
parser.add_argument("--runRankSVM", action="store_true", default=False)
parser.add_argument("--runRankBoost", action="store_true", default=False)
args = parser.parse_args()

helper = MetabolomicsExpHelper(dataDict, YCortisol, YTesto, YIgf1, ages)
helper.runCartTreeRank = args.runCartTR
helper.runRbfSvmTreeRank = args.runRbfSvmTR
helper.runL1SvmTreeRank = args.runL1SvmTR
helper.runCartTreeRankForest = args.runCartTRF
helper.runRbfSvmTreeRankForest = args.runRbfSvmTRF
helper.runL1SvmTreeRankForest = args.runL1SvmTRF
helper.runRankSVM = args.runRankSVM
helper.runRankBoost = args.runRankBoost
helper.run()
