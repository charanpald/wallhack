import numpy
import logging
import sys
import argparse 
from wallhack.rankingexp.RankingExpHelper import RankingExpHelper
from wallhack.rankingexp.DatasetUtils import DatasetUtils
from sandbox.util.Util import Util 
Util.setupScript()

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.alphas = numpy.array([0.0625])
defaultAlgoArgs.folds = 1
defaultAlgoArgs.ks = numpy.array([64])
defaultAlgoArgs.lmbdasMlauc = 2.0**-numpy.arange(1, 8)
defaultAlgoArgs.numRowSamples = 15
defaultAlgoArgs.parallelSGD = True
defaultAlgoArgs.recordFolds = 1
defaultAlgoArgs.validationUsers = 0.1

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RankingExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

#Create/load a low rank matrix 
X = DatasetUtils.flixster()
(m, n) = X.shape


dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "Flixster"

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
rankingExpHelper.runExperiment(X)