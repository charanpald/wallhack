import numpy
import logging
import sys
import argparse 
import os
import errno
from wallhack.rankingexp.DatasetUtils import DatasetUtils 
from wallhack.rankingexp.RankingExpHelper import RankingExpHelper
import sppy 
import sppy.io

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = 2**numpy.arange(4, 8)

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RankingExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

#Load/create the dataset 
X = DatasetUtils.mendeley()
m, n = X.shape

defaultAlgoArgs.u = 5/float(n) 
defaultAlgoArgs.maxIterations = 30 
defaultAlgoArgs.numRowSamples = 10 

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "MendeleyCoauthors"

# print args #
logging.info("Running on " + dataArgs.extendedDirName)
logging.info("Data params:")
keys = list(vars(dataArgs).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(dataArgs.__getattribute__(key)))

logging.info("Creating the exp-runner")
rankingExpHelper = RankingExpHelper(remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
rankingExpHelper.printAlgoArgs()
#    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
try:
    os.makedirs(rankingExpHelper.resultsDir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

rankingExpHelper.runExperiment(X)
